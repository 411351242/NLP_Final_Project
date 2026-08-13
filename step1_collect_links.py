import asyncio
import sys
import json
import os
import re
import random
import argparse
import pandas as pd
from typing import List, Set, Tuple
from playwright.async_api import async_playwright, Page

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


async def is_container_pinned(container) -> bool:
    """
    判斷該貼文區塊是否為置頂貼文 (Pinned Post)。
    檢查方式：尋找 svg 圖示的 aria-label 是否包含 pin/置頂，或前幾行文字是否包含置頂字樣。
    """
    try:
        svg_labels = await container.locator('svg').evaluate_all(
            "elements => elements.map(e => e.getAttribute('aria-label') || '')"
        )
        for label in svg_labels:
            if re.search(r'pin|置頂', label, re.IGNORECASE):
                return True

        inner_text = await container.inner_text()
        first_few_lines = inner_text.split('\n')[:4]
        for line in first_few_lines:
            line_clean = line.strip().lower()
            if line_clean in {"pinned", "已置頂", "置頂"}:
                return True
    except Exception:
        pass
    return False


def clean_post_url(href: str, target_username: str) -> str:
    """
    將爬取到的相對路徑或帶有參數的 URL 標準化為乾淨的貼文網址。
    格式: https://www.threads.net/@{username}/post/{post_id}
    """
    if not href:
        return ""
    full_url = href if href.startswith("http") else f"https://www.threads.net{href}"
    full_url = full_url.split("?")[0]

    if "/post/" in full_url:
        base_part, post_part = full_url.split("/post/", 1)
        post_id = post_part.split("/")[0]
        return f"https://www.threads.net/@{target_username}/post/{post_id}"
    return full_url


async def collect_links(
    page: Page,
    target_username: str,
    existing_urls_set: Set[str],
    early_stop_threshold: int = 4
) -> Tuple[List[str], int]:
    """
    滾動目標 Threads 主頁並收集貼文網址。
    
    支援增量檢查與早停機制：
    若連續遇到已存在於 existing_urls_set 的舊貼文達到 early_stop_threshold 篇，
    即判定已銜接歷史資料，提前停止滾動以節省時間。
    
    :param page: Playwright Page 物件
    :param target_username: 目標使用者帳號
    :param existing_urls_set: 已知的歷史貼文網址集合
    :param early_stop_threshold: 連續命中舊貼文的早停門檻 (預設 4 篇)
    :return: (本次新發現之貼文網址列表, 新發現貼文數量)
    """
    target_url = f"https://www.threads.net/@{target_username}"
    print(f"[*] 正在前往目標主頁：{target_url}", flush=True)
    await page.goto(target_url, wait_until='domcontentloaded')

    if "/login" in page.url.lower():
        print("[!] 提示：偵測到頁面跳轉至登入頁，請確認 cookies.json 是否有效。", flush=True)

    try:
        await page.wait_for_selector(f'a[href*="/@{target_username}/post/"]', timeout=15000)
    except Exception:
        print("[!] 提示：未在頁面中找到貼文連結，可能頁面尚未載入完全或帳號為私密。", flush=True)

    await asyncio.sleep(random.uniform(1.5, 2.5))

    new_discovered_urls: dict = {}
    seen_in_session: Set[str] = set()
    consecutive_known_count = 0
    early_stopped = False

    no_change_count = 0
    max_no_change = 4
    current_scroll = 0
    viewport_height = 800

    print("[*] 開始滾動頁面檢索貼文...", flush=True)
    if existing_urls_set:
        print(f"[*] 增量更新模式已啟動：已載入 {len(existing_urls_set)} 筆歷史貼文，連續命中 {early_stop_threshold} 篇舊貼文將自動早停。", flush=True)
    else:
        print("[*] 全量爬取模式啟動：未發現歷史貼文記錄，將滾動至最底端。", flush=True)

    while no_change_count < max_no_change and not early_stopped:
        current_height = await page.evaluate("document.body.scrollHeight")

        while current_scroll < current_height:
            current_scroll += viewport_height
            target_scroll = min(current_scroll, current_height)
            await page.evaluate(f"window.scrollTo(0, {target_scroll})")
            await asyncio.sleep(random.uniform(0.4, 0.7))

            containers = await page.locator("article, div[data-pressable-container='true']").all()
            for container in containers:
                try:
                    href_element = container.locator(f'a[href*="/@{target_username}/post/"]')
                    count = await href_element.count()
                    if count == 0:
                        continue

                    raw_href = await href_element.first.get_attribute("href")
                    clean_url = clean_post_url(raw_href, target_username)
                    if not clean_url or clean_url in seen_in_session:
                        continue

                    seen_in_session.add(clean_url)
                    is_pinned = await is_container_pinned(container)

                    if is_pinned:
                        # 置頂貼文不計入連續舊貼文計數
                        print(f"   [置頂貼文] 發現置頂貼文: {clean_url}", flush=True)
                        if clean_url not in existing_urls_set:
                            new_discovered_urls[clean_url] = None
                    else:
                        if clean_url in existing_urls_set:
                            consecutive_known_count += 1
                            print(f"   [歷史貼文] 命中已存在貼文 ({consecutive_known_count}/{early_stop_threshold}): {clean_url}", flush=True)
                            if consecutive_known_count >= early_stop_threshold:
                                print(f"[*] 連續命中 {consecutive_known_count} 篇歷史貼文，已銜接前次進度，觸發早停 (Early-Stop)。", flush=True)
                                early_stopped = True
                                break
                        else:
                            consecutive_known_count = 0
                            new_discovered_urls[clean_url] = None
                            print(f"   [新貼文] 發現新貼文: {clean_url}", flush=True)
                except Exception:
                    continue

            if early_stopped:
                break

        if early_stopped:
            break

        # 模擬滾動到底部
        await page.keyboard.press("End")
        await asyncio.sleep(random.uniform(1.8, 2.4))

        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == current_height:
            await page.evaluate("window.scrollBy(0, -600)")
            await asyncio.sleep(0.5)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1.5)

            final_height = await page.evaluate("document.body.scrollHeight")
            if final_height == current_height:
                no_change_count += 1
            else:
                no_change_count = 0
                current_height = final_height
        else:
            no_change_count = 0

    new_count = len(new_discovered_urls)
    print(f"[*] 貼文檢索完成，本次發現新貼文 {new_count} 篇。", flush=True)
    return list(new_discovered_urls.keys()), new_count


async def main():
    parser = argparse.ArgumentParser(description="Threads 貼文連結收集與增量更新工具 (Step 1)")
    parser.add_argument("--user", "-u", default="make_investment_easy", help="目標 Threads 使用者帳號 (預設: make_investment_easy)")
    parser.add_argument("--cookie", default="cookies.json", help="Cookie 檔案路徑 (預設: cookies.json)")
    parser.add_argument("--threshold", type=int, default=4, help="連續命中歷史貼文的早停門檻 (預設: 4)")
    args = parser.parse_args()

    target_username = args.user.lstrip("@").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "threads_post_links.csv")
    cookie_file = os.path.join(script_dir, args.cookie)

    existing_urls_list = []
    existing_urls_set = set()
    if os.path.exists(output_file):
        try:
            df_old = pd.read_csv(output_file, encoding="utf-8-sig")
            if "貼文網址" in df_old.columns:
                existing_urls_list = df_old["貼文網址"].dropna().tolist()
                existing_urls_set = set(existing_urls_list)
                print(f"[*] 載入現有記錄：現有 {len(existing_urls_set)} 筆貼文連結。", flush=True)
        except Exception as e:
            print(f"[!] 讀取歷史檔案失敗: {e}", flush=True)

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
        new_links, new_count = await collect_links(
            page,
            target_username,
            existing_urls_set,
            early_stop_threshold=args.threshold
        )

        combined_links = new_links + [u for u in existing_urls_list if u not in new_links]

        if combined_links:
            df = pd.DataFrame(combined_links, columns=["貼文網址"])
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[*] 連結清單更新完成：{output_file} (總計: {len(combined_links)} 篇，新增: {new_count} 篇)", flush=True)
        else:
            print("[!] 未收集到任何貼文連結。", flush=True)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
