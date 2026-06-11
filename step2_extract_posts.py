import asyncio
import json
import os
import math
import glob
import re
import pandas as pd
from typing import List, Dict, Set
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup

def is_count_string(s: str) -> bool:
    """
    檢查是否為互動數據 (例如：185, 2K, 1.2M, 2萬)
    """
    s_clean = s.replace(',', '').strip()
    if not s_clean:
        return False
    # 匹配純數字或帶有 K, M, B, 萬 等後綴的數字
    return bool(re.match(r'^\d+(\.\d+)?[kKmMgGbB萬]?$', s_clean))

def is_datetime_string(s: str) -> bool:
    """
    檢查是否為時間/日期格式
    """
    s = s.strip()
    if re.match(r'^\d+[dhms]$', s): # e.g. 3d, 17h
        return True
    if re.match(r'^\d{2,4}[-/\\]\d{1,2}[-/\\]\d{1,4}$', s): # e.g. 04/22/26, 2025-5-14
        return True
    if s in {"·", "•"}:
        return True
    return False

def clean_post_text(inner_text: str, target_username: str) -> str:
    """
    清洗貼文內文，去除 UI 雜訊 (作者名、日期、按讚、回覆、分享等字眼以及純數字的互動數據)
    """
    lines = inner_text.split('\n')
    lines = [l.strip() for l in lines]
    # 過濾空行
    lines = [l for l in lines if l]
    
    # 1. 清除頂部的標籤與中繼資料 (Pinned, Author, @username, 時間)
    header_keywords = {"pinned", "author", "·", "•", target_username.lower()}
    
    start_idx = 0
    while start_idx < len(lines):
        line_lower = lines[start_idx].lower()
        if line_lower in header_keywords:
            start_idx += 1
        elif is_datetime_string(line_lower):
            start_idx += 1
        else:
            break
            
    lines = lines[start_idx:]
        
    cleaned_lines = []
    # 常見的 Threads 動作與按鈕關鍵字 (中英文) 以及輪播圖指標
    skip_words = {
        "translate", "like", "reply", "share", "likes", "replies", "shares",
        "翻譯", "讚", "回覆", "分享", "查看更多回覆", "view more replies",
        "top", "view activity", "view activityview activity",
        "/", "·", "•", "author", "pinned"
    }
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower() in skip_words:
            continue
        if is_count_string(line_stripped):
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def export_batch(data: List[Dict], batch_num: int):
    """
    匯出分批 CSV 檔案
    """
    if not data:
        return
    filename = f"threads_posts-{batch_num:02d}.csv"
    df = pd.DataFrame(data, columns=["貼文與串文編號", "文字內容"])
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"[+] 成功匯出批次檔案：{filename}")

async def scrape_post_detail(page: Page, url: str, target_username: str, post_index: int) -> tuple[List[Dict], List[str]]:
    """
    進入單篇貼文，抓取主貼文與該作者的後續串文，並回傳這些容器的 Post URL 以避免重複爬取
    """
    print(f"   -> 正在讀取: {url}")
    try:
        await page.goto(url, wait_until='domcontentloaded')
        # 等待貼文容器載入
        try:
            await page.wait_for_selector("article, div[data-pressable-container='true']", timeout=10000)
        except Exception:
            pass
            
        # 稍微向下捲動讓串文載入
        for _ in range(2):
            await page.evaluate("window.scrollBy(0, 500)")
            await asyncio.sleep(1.5)
            
        # 尋找所有的貼文容器
        containers = await page.locator("article, div[data-pressable-container='true']").all()
        if not containers:
            return [], []
            
        results = []
        scraped_urls = []
        seen_post_ids = set() # 用於同一頁面防重覆
        thread_part = 1
        for container in containers:
            try:
                html = await container.inner_html()
            except Exception:
                continue
                
            soup = BeautifulSoup(html, "html.parser")
            
            # 尋找作者名稱 (通常是 href="/@username")
            links = soup.find_all("a", href=True)
            authors = [a['href'] for a in links if '/@' in a['href'] and '/post/' not in a['href']]
            
            container_author = None
            if authors:
                container_author = authors[0].replace("/@", "").split("?")[0]
            else:
                # 備用方案：如果沒有找到帳號連結，檢查是否有帶有 /post/ 且含有帳號的連結
                authors_from_post = [a['href'].split('/post/')[0].replace('/@', '').split('?')[0] for a in links if '/post/' in a['href'] and '/@' in a['href']]
                if authors_from_post:
                    container_author = authors_from_post[0]
            
            if not container_author:
                continue
                
            if container_author.lower() != target_username.lower():
                # 若遇到不是目標作者的容器，代表這已經是其他人的回覆，或者是前方的「他人主貼文」（代表目標使用者在回覆他人），直接停止本頁爬取
                print(f"      [info] 遇到非目標作者：@{container_author}，停止本頁爬取。")
                break
                
            # 提取這個容器的 Post ID 以免重複抓取同頁的嵌套容器
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
                
            # 提取 innerText
            try:
                inner_text = await container.inner_text()
            except Exception:
                continue
                
            # 清洗內文
            cleaned_content = clean_post_text(inner_text, target_username)
            
            if cleaned_content.strip():
                results.append({
                    "貼文與串文編號": f"貼文{post_index}_第{thread_part}串",
                    "文字內容": cleaned_content
                })
                thread_part += 1
                
        return results, scraped_urls
    except Exception as e:
        print(f"[!] 爬取 {url} 時發生錯誤: {e}")
        return [], []

async def main():
    target_username = "make_investment_easy"
    cookie_file = "cookies.json"
    input_file = "threads_post_links.csv"
    batch_size = 50
    
    if not os.path.exists(input_file):
        print(f"[!] 找不到輸入檔案 {input_file}，請先執行步驟一腳本。")
        return
        
    df_links = pd.read_csv(input_file)
    all_urls = df_links['貼文網址'].tolist()
    
    processed_urls = set()
    
    print(f"[*] 總共收集到 {len(all_urls)} 篇貼文 URL。")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        
        if os.path.exists(cookie_file):
            print(f"[*] 載入登入憑證 {cookie_file}...")
            with open(cookie_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)
                await context.add_cookies(cookies)
                
        page = await context.new_page()
        
        # 每次重跑都從 1 開始
        post_index = 1
        current_batch_num = 1
        print(f"[*] 貼文編號起點：{post_index}")
        
        current_batch_data = []
        urls_in_current_batch = 0
        
        for url in all_urls:
            # 確保 URL 格式乾淨
            if "/post/" not in url:
                continue
            base_part, post_part = url.split("/post/", 1)
            post_id = post_part.split("/")[0].split("?")[0]
            clean_url = f"https://www.threads.net/@{target_username}/post/{post_id}"
            
            if clean_url in processed_urls:
                continue
                
            data, scraped_urls_batch = await scrape_post_detail(page, clean_url, target_username, post_index)
            if data:
                current_batch_data.extend(data)
                
            # 將該串貼文所有偵測到的主貼文與串文 URLs 都標記為已處理
            for s_url in scraped_urls_batch:
                processed_urls.add(s_url)
            processed_urls.add(clean_url)
            
            urls_in_current_batch += 1
            post_index += 1
            
            # 若滿足 batch 大小，則匯出
            if urls_in_current_batch >= batch_size:
                export_batch(current_batch_data, current_batch_num)
                current_batch_data = []
                urls_in_current_batch = 0
                current_batch_num += 1
                
        # 迴圈結束後，若有剩餘資料未匯出
        if current_batch_data:
            export_batch(current_batch_data, current_batch_num)
            
        await browser.close()
        
    print("[*] 第二階段爬取完畢！")

if __name__ == "__main__":
    asyncio.run(main())
