import asyncio
import json
import os
import pandas as pd
from typing import List, Set
from playwright.async_api import async_playwright, Page

async def collect_links(page: Page, target_username: str) -> List[str]:
    target_url = f"https://www.threads.net/@{target_username}"
    print(f"[*] 正在前往目標主頁：{target_url}")
    await page.goto(target_url, wait_until='domcontentloaded')
    try:
        await page.wait_for_selector(f'a[href*="/@{target_username}/post/"]', timeout=15000)
    except Exception:
        print("[!] 警告：未在頁面中找到貼文連結，可能頁面尚未載入完全、該用戶無貼文，或登入憑證失效。")
    await asyncio.sleep(2) # 稍微等待額外內容渲染
    
    links: dict = {}
    previous_height = await page.evaluate("document.body.scrollHeight")
    
    no_change_count = 0
    max_no_change = 10  # 增加容忍次數，確保能跑到最底
    
    while no_change_count < max_no_change:
        # 漸進式向下滾動，每次滾動 1000px 並即時收集連結，避免 Virtual DOM 卸載中間的節點
        current_scroll = 0
        viewport_height = 1000
        
        while current_scroll < previous_height:
            current_scroll = min(current_scroll + viewport_height, previous_height)
            await page.evaluate(f"window.scrollTo(0, {current_scroll})")
            await asyncio.sleep(0.5) # 稍微等待 Virtual DOM 渲染
            
            # 即時尋找 a[href*="/@username/post/"] 的元素
            hrefs = await page.locator(f'a[href*="/@{target_username}/post/"]').evaluate_all(
                "elements => elements.map(e => e.getAttribute('href'))"
            )
            for href in hrefs:
                if href:
                    full_url = href if href.startswith("http") else f"https://www.threads.net{href}"
                    full_url = full_url.split("?")[0]  # 去除 query string
                    
                    # 清除 post ID 後面的 /media
                    if "/post/" in full_url:
                        base_part, post_part = full_url.split("/post/", 1)
                        post_id = post_part.split("/")[0]
                        full_url = f"{base_part}/post/{post_id}"
                    
                    links[full_url] = None
                
        # 觸發滾動以加載更多內容
        await page.keyboard.press("End")
        await asyncio.sleep(3) # 增加等待時間讓伺服器回應
        
        current_height = await page.evaluate("document.body.scrollHeight")
        if current_height == previous_height:
            # 如果高度沒變，稍微往上滾再往下滾，強迫觸發 IntersectionObserver
            await page.evaluate("window.scrollBy(0, -1000)")
            await asyncio.sleep(1)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)
            
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                no_change_count += 1
            else:
                no_change_count = 0
                previous_height = new_height
        else:
            no_change_count = 0
            previous_height = current_height
            
        print(f"   目前已抓取 {len(links)} 個貼文連結... (閒置計數: {no_change_count}/{max_no_change})")
        
    print(f"[*] 連結收集完成！總共收集到 {len(links)} 篇貼文連結。")
    return list(links.keys())

async def main():
    target_username = "make_investment_easy"
    cookie_file = "cookies.json"
    output_file = "threads_post_links.csv"
    
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
        else:
            print(f"[!] 警告：找不到 {cookie_file}，將以訪客身份執行")
            
        page = await context.new_page()
        post_links = await collect_links(page, target_username)
        
        if post_links:
            df = pd.DataFrame(post_links, columns=["貼文網址"])
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[*] 連結已成功匯出至：{output_file}")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
