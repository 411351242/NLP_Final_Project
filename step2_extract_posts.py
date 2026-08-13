import asyncio
import sys
import json
import os
import re
import random
import argparse
import pandas as pd
from typing import List, Dict, Set, Tuple
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def clean_post_text(html_content: str, target_username: str) -> str:
    """
    解析貼文 HTML 並過濾非正文的 UI 元件與平台字詞。
    
    過濾項目：
    1. 作者名稱、按讚、轉發、回覆、翻譯、分享等固定字詞。
    2. 時間標籤 (如 12h, 3d, 2024-05-01)。
    3. 數字互動數據 (如 1.2k, 500)。
    4. 回覆作者的前置提示文字 (Reply to username...)。
    
    :param html_content: 貼文區塊的 HTML 內容
    :param target_username: 目標使用者帳號
    :return: 清洗後的正文字串
    """
    soup = BeautifulSoup(html_content, "html.parser")
    spans = soup.find_all("span", dir="auto")

    cleaned_lines = []
    exact_skip_words = {
        "translate", "like", "reply", "share", "likes", "replies", "shares",
        "翻譯", "讚", "回覆", "分享", "查看更多回覆", "view more replies",
        "top", "view activity", "author", "pinned", "·", "•", "read more"
    }

    for span in spans:
        text = span.get_text(strip=True)
        if not text:
            continue

        text_lower = text.lower()
        if text_lower == target_username.lower():
            continue
        if text_lower in exact_skip_words:
            continue
        if text_lower == "view activityview activity":
            continue
        if re.match(r'^\d{2,4}[-/\]\d{1,2}[-/\]\d{1,4}$', text):
            continue
        if re.match(r'^\d+[dhms]$', text):
            continue
        if re.match(r'^\d+(\.\d+)?[kKmMgGbB萬]?$', text.replace(',', '')):
            continue

        text = re.sub(r'Translate\d+/\d+$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Translate$', '', text, flags=re.IGNORECASE)

        if re.match(rf'^Reply to {target_username.lower()}\.\.\.$', text_lower):
            continue

        if text:
            cleaned_lines.append(text)

    return "\n".join(cleaned_lines)


async def scrape_post_detail(
    page: Page,
    url: str,
    target_username: str,
    post_index: int
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    進入單篇貼文詳細頁面，滾動並提取原作者的主貼文與後續串文。
    遇到其他非目標作者的回覆即停止該篇向下爬取。
    
    :param page: Playwright Page 物件
    :param url: 目標貼文網址
    :param target_username: 目標作者帳號
    :param post_index: 貼文編號
    :return: (提取之貼文資料列表, 本次解析到的所有子貼文網址列表)
    """
    try:
        await page.goto(url, wait_until='domcontentloaded')

        if "/login" in page.url.lower():
            print("[!] 提示：偵測到頁面跳轉至登入頁，請確認 cookies.json 是否有效。", flush=True)

        try:
            await page.wait_for_selector("article, div[data-pressable-container='true']", timeout=10000)
        except Exception:
            return [], []

        results = []
        scraped_urls = []
        seen_post_ids = set()
        thread_part = 1

        last_container_count = 0
        scroll_attempts = 0
        max_scroll_attempts = 15

        while scroll_attempts < max_scroll_attempts:
            containers = await page.locator("article, div[data-pressable-container='true']").all()
            if not containers:
                await asyncio.sleep(random.uniform(0.6, 1.0))
                scroll_attempts += 1
                continue

            hit_other_author = False

            for container in containers[last_container_count:]:
                try:
                    html = await container.inner_html()
                except Exception:
                    continue

                soup = BeautifulSoup(html, "html.parser")
                links = soup.find_all("a", href=True)
                authors = [a['href'] for a in links if '/@' in a['href'] and '/post/' not in a['href']]

                container_author = None
                if authors:
                    container_author = authors[0].replace("/@", "").split("?")[0]
                else:
                    authors_from_post = [
                        a['href'].split('/post/')[0].replace('/@', '').split('?')[0]
                        for a in links if '/post/' in a['href'] and '/@' in a['href']
                    ]
                    if authors_from_post:
                        container_author = authors_from_post[0]

                if not container_author:
                    continue

                # 遇到非目標作者留言即終止串文向下解析
                if container_author.lower() != target_username.lower():
                    hit_other_author = True
                    break

                container_post_id = None
                for a in links:
                    href = a.get('href', '')
                    if "/post/" in href:
                        container_post_id = href.split("/post/")[1].split("/")[0].split("?")[0]
                        break

                if container_post_id:
                    if container_post_id in seen_post_ids:
                        continue
                    seen_post_ids.add(container_post_id)
                    clean_url = f"https://www.threads.net/@{target_username}/post/{container_post_id}"
                    scraped_urls.append(clean_url)

                cleaned_content = clean_post_text(html, target_username)
                if cleaned_content.strip():
                    results.append({
                        "貼文與串文編號": f"貼文{post_index}_第{thread_part}串",
                        "文字內容": cleaned_content
                    })
                    thread_part += 1

            if hit_other_author:
                break

            last_container_count = len(containers)
            await page.evaluate("window.scrollBy(0, 800)")
            await asyncio.sleep(random.uniform(0.6, 1.0))

            new_containers_count = await page.locator("article, div[data-pressable-container='true']").count()
            if new_containers_count == last_container_count:
                await asyncio.sleep(1)
                final_count = await page.locator("article, div[data-pressable-container='true']").count()
                if final_count == last_container_count:
                    break

            scroll_attempts += 1

        return results, scraped_urls
    except Exception as e:
        print(f"[!] 爬取 {url} 時發生錯誤: {e}", flush=True)
        return [], []


def get_next_post_index(posts_file: str) -> int:
    """
    讀取現有貼文 CSV 檔案，取得下一篇可用的貼文序號。
    """
    if not os.path.exists(posts_file):
        return 1
    try:
        df = pd.read_csv(posts_file, encoding="utf-8-sig")
        if "貼文與串文編號" not in df.columns or df.empty:
            return 1

        post_nums = []
        for val in df["貼文與串文編號"].dropna():
            match = re.search(r'貼文(\d+)', str(val))
            if match:
                post_nums.append(int(match.group(1)))
        if post_nums:
            return max(post_nums) + 1
    except Exception:
        pass
    return 1


async def main():
    parser = argparse.ArgumentParser(description="Threads 貼文與串文詳細內文提取工具 (Step 2)")
    parser.add_argument("--user", "-u", default="make_investment_easy", help="目標 Threads 使用者帳號 (預設: make_investment_easy)")
    parser.add_argument("--cookie", default="cookies.json", help="Cookie 檔案路徑")
    args = parser.parse_args()

    target_username = args.user.lstrip("@").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    links_file = os.path.join(script_dir, "threads_post_links.csv")
    posts_file = os.path.join(script_dir, "threads_posts.csv")
    processed_urls_file = os.path.join(script_dir, "processed_urls.json")
    cookie_file = os.path.join(script_dir, args.cookie)

    if not os.path.exists(links_file):
        print(f"[!] 找不到連結清單 {links_file}，請先執行 step1_collect_links.py。", flush=True)
        return

    df_links = pd.read_csv(links_file, encoding="utf-8-sig")
    all_urls = df_links["貼文網址"].dropna().tolist()

    # 讀取已處理的 URL 記錄，支援斷點續爬
    processed_urls = set()
    if os.path.exists(processed_urls_file):
        try:
            with open(processed_urls_file, "r", encoding="utf-8") as f:
                processed_urls = set(json.load(f))
            print(f"[*] 載入斷點續爬記錄：已處理 {len(processed_urls)} 個 URL。", flush=True)
        except Exception as e:
            print(f"[!] 讀取斷點記錄失敗: {e}", flush=True)

    pending_urls = [u for u in all_urls if u not in processed_urls]
    print(f"[*] 總貼文連結數：{len(all_urls)}，待處理新貼文：{len(pending_urls)}", flush=True)

    if not pending_urls:
        print("[*] 所有貼文皆已抓取完成，無需額外提取。", flush=True)
        return

    post_index = get_next_post_index(posts_file)
    print(f"[*] 貼文編號將從第 {post_index} 篇開始接續編寫。", flush=True)

    if not os.path.exists(posts_file):
        df_init = pd.DataFrame(columns=["貼文與串文編號", "文字內容"])
        df_init.to_csv(posts_file, index=False, encoding="utf-8-sig")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )

        if os.path.exists(cookie_file):
            print(f"[*] 載入登入憑證 {cookie_file}...", flush=True)
            with open(cookie_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)
                await context.add_cookies(cookies)
        else:
            print(f"[!] 提示：找不到 {cookie_file}，將以訪客身份執行", flush=True)

        page = await context.new_page()

        for idx, url in enumerate(pending_urls, start=1):
            if "/post/" not in url:
                continue
            base_part, post_part = url.split("/post/", 1)
            post_id = post_part.split("/")[0].split("?")[0]
            clean_url = f"https://www.threads.net/@{target_username}/post/{post_id}"

            if clean_url in processed_urls:
                continue

            print(f"[{idx}/{len(pending_urls)}] 正在讀取: {clean_url}", flush=True)
            data, scraped_urls_batch = await scrape_post_detail(page, clean_url, target_username, post_index)

            # 每成功提取一篇即時寫入 CSV，避免中斷造成資料損失
            if data:
                df_batch = pd.DataFrame(data, columns=["貼文與串文編號", "文字內容"])
                df_batch.to_csv(posts_file, mode="a", header=False, index=False, encoding="utf-8-sig")
                print(f"   -> 成功提取 {len(data)} 則串文，已寫入 {posts_file} (貼文{post_index})", flush=True)
                post_index += 1
            else:
                print(f"   -> 未找到有效內文或非目標作者", flush=True)

            for s_url in scraped_urls_batch:
                processed_urls.add(s_url)
            processed_urls.add(clean_url)

            # 同步更新斷點紀錄檔
            with open(processed_urls_file, "w", encoding="utf-8") as f:
                json.dump(list(processed_urls), f, ensure_ascii=False, indent=2)

        await browser.close()

    print(f"[*] 第二階段提取完成，所有內容已儲存至：{posts_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
