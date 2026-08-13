import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from step1_collect_links import main as step1_main
from step2_extract_posts import main as step2_main
from process_threads_by_post import process_csv_threads


def generate_summary_report(
    output_txt_path: str,
    username: str,
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    total_time: float,
    new_links_count: int,
    extracted_posts_count: int,
    total_posts: int,
    total_threads: int,
    total_processed_urls: int,
    stage2_skipped: bool,
    stage3_skipped: bool
) -> str:
    """
    生成並儲存資料管線執行結果的文字摘要報告。
    
    :param output_txt_path: 摘要報告輸出路徑
    :param username: 目標使用者帳號
    :param stage1_time: 階段一執行耗時 (秒)
    :param stage2_time: 階段二執行耗時 (秒)
    :param stage3_time: 階段三執行耗時 (秒)
    :param total_time: 總執行時間 (秒)
    :param new_links_count: 本次新增的連結數量
    :param extracted_posts_count: 本次提取的貼文數量
    :param total_posts: 知識庫總貼文數量
    :param total_threads: 原始串文總數量
    :param total_processed_urls: 累計已處理的 URL 數量
    :param stage2_skipped: 階段二是否跳過
    :param stage3_skipped: 階段三是否跳過
    :return: 報告內文字串
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stage2_desc = f"{stage2_time:.2f} 秒 (無新貼文，已自動跳過)" if stage2_skipped else f"{stage2_time:.2f} 秒"
    stage3_desc = f"{stage3_time:.2f} 秒 (無資料變動，已自動跳過)" if stage3_skipped else f"{stage3_time:.2f} 秒"

    report = f"""============================================================
Threads 資料管線更新摘要報告 (Pipeline Summary Report)
============================================================
更新時間: {timestamp_str}
目標帳號: @{username}
執行狀態: 成功完成 (Success)

------------------------------------------------------------
各階段執行耗時 (Execution Time)
------------------------------------------------------------
[階段 1] 連結收集與增量檢查 : {stage1_time:.2f} 秒
[階段 2] 內文與串文提取     : {stage2_desc}
[階段 3] 長文重組與向量建庫 : {stage3_desc}
[總計耗時] 總執行時間       : {total_time:.2f} 秒

------------------------------------------------------------
數據統計與更新量 (Data & Update Statistics)
------------------------------------------------------------
* 本次新發現連結數 (New Links)       : {new_links_count} 篇
* 本次新提取貼文數 (New Extracted)   : {extracted_posts_count} 篇
* 知識庫長篇貼文數 (Total Posts)     : {total_posts} 篇
* 知識庫原始串文數 (Total Threads)   : {total_threads} 則
* 累積已處理 URL 數 (Processed URLs) : {total_processed_urls} 個

------------------------------------------------------------
輸出檔案與快取狀態 (Artifacts & Cache)
------------------------------------------------------------
* 貼文連結清單   : threads_post_links.csv
* 原始串文資料庫 : threads_posts.csv
* 結構化長文庫   : combined_threads_posts.csv
* 向量快取索引   : embeddings_index.pkl
* 中繼資料設定   : pipeline_metadata.json
* 摘要報告檔案   : {os.path.basename(output_txt_path)}
============================================================
"""
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[*] 摘要報告已生成並存檔至：{output_txt_path}", flush=True)
    return report


async def run_pipeline(target_username: str, cookie_file: str = "cookies.json", threshold: int = 4):
    """
    執行完整自動化資料處理流水線：
    1. 執行 step1_collect_links.py 增量收集目標帳號的最新貼文網址。
    2. 檢查未處理的 URL，若有新貼文則執行 step2_extract_posts.py 進行內文爬取與斷點續存。
    3. 若有新增資料或索引遺失，執行 process_threads_by_post.py 進行對話鏈重組與向量索引更新。
    4. 輸出執行耗時與統計摘要報告。
    
    :param target_username: 目標 Threads 帳號
    :param cookie_file: 登入憑證 JSON 檔案路徑
    :param threshold: 連續命中已存在貼文時的早停門檻
    """
    overall_start_time = time.time()
    username = target_username.lstrip("@").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    links_file = os.path.join(script_dir, "threads_post_links.csv")
    posts_file = os.path.join(script_dir, "threads_posts.csv")
    combined_file = os.path.join(script_dir, "combined_threads_posts.csv")
    processed_urls_file = os.path.join(script_dir, "processed_urls.json")
    summary_txt_file = os.path.join(script_dir, "pipeline_summary.txt")

    print("\n" + "="*60, flush=True)
    print(f"開始執行 Threads 資料管線處理：@{username}", flush=True)
    print("="*60, flush=True)

    # 記錄執行前的歷史連結數量
    initial_links_count = 0
    if os.path.exists(links_file):
        try:
            df_old = pd.read_csv(links_file, encoding="utf-8-sig")
            initial_links_count = len(df_old)
        except Exception:
            pass

    # ----------------------------------------------------
    # 步驟 1：增量收集貼文 URL
    # ----------------------------------------------------
    print(f"\n[階段 1/3] 檢查並收集貼文連結 (增量模式)...", flush=True)
    s1_start = time.time()

    sys.argv = [
        "step1_collect_links.py",
        "--user", username,
        "--cookie", cookie_file,
        "--threshold", str(threshold)
    ]
    await step1_main()
    stage1_time = time.time() - s1_start

    if not os.path.exists(links_file):
        print(f"[!] 錯誤：未能生成連結清單 {links_file}，流程終止。", flush=True)
        return

    df_links = pd.read_csv(links_file, encoding="utf-8-sig")
    all_urls = df_links["貼文網址"].dropna().tolist()
    new_links_count = max(0, len(all_urls) - initial_links_count)

    # 檢查尚未提取內文的貼文
    processed_urls = set()
    if os.path.exists(processed_urls_file):
        try:
            with open(processed_urls_file, "r", encoding="utf-8") as f:
                processed_urls = set(json.load(f))
        except Exception:
            pass

    pending_urls = [u for u in all_urls if u not in processed_urls]

    # ----------------------------------------------------
    # 步驟 2：提取內文與串文
    # ----------------------------------------------------
    print(f"\n[階段 2/3] 提取貼文與串文內容...", flush=True)
    s2_start = time.time()
    stage2_skipped = False
    extracted_posts_count = len(pending_urls)

    if pending_urls:
        print(f"[*] 發現 {len(pending_urls)} 篇新貼文待提取內文，啟動 Playwright 爬蟲...", flush=True)
        sys.argv = [
            "step2_extract_posts.py",
            "--user", username,
            "--cookie", cookie_file
        ]
        await step2_main()
    else:
        stage2_skipped = True
        print(f"[*] 所有貼文內文皆已是最新狀態 (共 {len(all_urls)} 篇)，跳過內文爬取。", flush=True)
    stage2_time = time.time() - s2_start

    # ----------------------------------------------------
    # 步驟 3：串文結構化重組與向量索引更新
    # ----------------------------------------------------
    print(f"\n[階段 3/3] 對話鏈結構化重組與向量索引更新...", flush=True)
    s3_start = time.time()
    stage3_skipped = False

    if os.path.exists(posts_file):
        embeddings_pkl = os.path.join(script_dir, "embeddings_index.pkl")
        need_rebuild = (extracted_posts_count > 0) or (not os.path.exists(embeddings_pkl)) or (not os.path.exists(combined_file))

        if need_rebuild:
            process_csv_threads(posts_file, combined_file)
        else:
            stage3_skipped = True
            print(f"[*] 向量索引與長文庫已是最新狀態，跳過重複運算。", flush=True)
    else:
        print(f"[!] 警告：找不到 {posts_file}，無法執行合併步驟。", flush=True)
    stage3_time = time.time() - s3_start

    total_time = time.time() - overall_start_time

    # 統計當前總篇數
    total_posts = 0
    total_threads = 0
    if os.path.exists(combined_file):
        try:
            total_posts = len(pd.read_csv(combined_file, encoding="utf-8-sig"))
        except Exception:
            pass
    if os.path.exists(posts_file):
        try:
            total_threads = len(pd.read_csv(posts_file, encoding="utf-8-sig"))
        except Exception:
            pass

    # 重新讀取最新的 processed_urls 計數
    if os.path.exists(processed_urls_file):
        try:
            with open(processed_urls_file, "r", encoding="utf-8") as f:
                processed_urls = set(json.load(f))
        except Exception:
            pass

    # 生成並儲存文字摘要報告
    report_text = generate_summary_report(
        output_txt_path=summary_txt_file,
        username=username,
        stage1_time=stage1_time,
        stage2_time=stage2_time,
        stage3_time=stage3_time,
        total_time=total_time,
        new_links_count=new_links_count,
        extracted_posts_count=extracted_posts_count if not stage2_skipped else 0,
        total_posts=total_posts,
        total_threads=total_threads,
        total_processed_urls=len(processed_urls),
        stage2_skipped=stage2_skipped,
        stage3_skipped=stage3_skipped
    )

    print("\n" + report_text, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Threads 貼文自動採集與處理流水線主程式 (Master Pipeline)")
    parser.add_argument("--user", "-u", default="make_investment_easy", help="目標 Threads 使用者帳號 (預設: make_investment_easy)")
    parser.add_argument("--cookie", default="cookies.json", help="Cookie 檔案路徑 (預設: cookies.json)")
    parser.add_argument("--threshold", type=int, default=4, help="連續命中舊貼文時的早停門檻 (預設: 4)")
    args = parser.parse_args()

    target_username = args.user.lstrip("@").strip()
    asyncio.run(run_pipeline(target_username, cookie_file=args.cookie, threshold=args.threshold))


if __name__ == "__main__":
    main()
