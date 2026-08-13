import os
import sys
import re
import argparse
import pickle
import json
from datetime import datetime
import pandas as pd
from rag_engine import TextCleaner, VectorIndexer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def clean_duplicates(df, text_column='文字內容', prefix_len=10):
    """
    去除重複貼文 (依貼文前 prefix_len 個字判斷)
    """
    temp_df = df.copy()
    temp_df['_feature'] = temp_df[text_column].str.strip().str.slice(0, prefix_len)
    is_duplicate = temp_df.duplicated(subset=['_feature'], keep='first')
    cleaned_df = temp_df[~is_duplicate].drop(columns=['_feature']).reset_index(drop=True)
    return cleaned_df

def process_csv_threads(csv_path: str, output_path: str):
    """
    讀取爬取到的 raw CSV 檔案，將多則串文按 Post ID 重組為單篇長文，
    並自動執行語意清洗、重新計算向量索引與更新中繼資料。
    """
    print(f"[*] 正在讀取 {csv_path}...", flush=True)

    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'cp950']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"[*] 成功使用 {encoding} 編碼讀取 CSV。", flush=True)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"無法讀取檔案 {csv_path}，請確認編碼是否為 utf-8 或 cp950。")

    required_cols = {'貼文與串文編號', '文字內容'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 檔案缺少必要欄位：{required_cols - set(df.columns)}")

    def parse_id(id_str):
        """解析 '貼文123_第2串' 或 '貼文123' 格式以取得數值序號"""
        if not isinstance(id_str, str):
            return 0, 0
        match = re.match(r'^貼文(\d+)_第(\d+)串$', id_str.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        match_single = re.match(r'^貼文(\d+)$', id_str.strip())
        if match_single:
            return int(match_single.group(1)), 1
        return 0, 0

    parsed = df['貼文與串文編號'].apply(parse_id)
    df['貼文數字'] = [p[0] for p in parsed]
    df['串順序'] = [p[1] for p in parsed]
    df = df.sort_values(by=['貼文數字', '串順序'])

    # 統一調用 TextCleaner 進行單篇串文初步清洗
    df['文字內容_clean'] = df['文字內容'].apply(TextCleaner.clean_thread_item)
    df['貼文編號'] = df['貼文數字'].apply(lambda x: f"貼文{x}")

    print("[*] 正在將同一貼文的多則串文合併為完整對話鏈...", flush=True)
    combined = df.groupby('貼文編號', sort=False)['文字內容_clean'].apply(
        lambda x: '\n'.join([s for s in x if s])
    ).reset_index()
    combined.rename(columns={'文字內容_clean': '文字內容'}, inplace=True)

    print("[*] 正在去除重複貼文...", flush=True)
    combined = clean_duplicates(combined)

    # 匯出結構化長文資料
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[*] ✅ 對話鏈合併與去重完成，共 {len(combined)} 篇貼文。已存檔至：{output_path}", flush=True)

    # 執行語意深度清洗並重建向量索引
    print("[*] 正在執行語意標準化並重新建構向量索引 (VectorIndexer)...", flush=True)
    combined['cleaned_text'] = combined['文字內容'].apply(TextCleaner.clean_text)
    combined = combined[combined['cleaned_text'].str.strip() != ''].copy()

    indexer = VectorIndexer()
    indexer.fit(combined['cleaned_text'], post_ids=combined['貼文編號'])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    indexer_pkl = os.path.join(script_dir, "embeddings_index.pkl")
    with open(indexer_pkl, "wb") as f:
        pickle.dump(indexer, f)
    print(f"[*] ✅ 向量索引已成功序列化並儲存至：{indexer_pkl}", flush=True)

    # 儲存中繼資料
    metadata = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_posts": len(combined),
        "total_threads": len(df)
    }
    metadata_json = os.path.join(script_dir, "pipeline_metadata.json")
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"[*] ✅ 統計中繼資料已儲存至：{metadata_json}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threads 串文結構化長文重組與向量索引建庫工具 (Step 3)")
    parser.add_argument("--user", "-u", default="make_investment_easy", help="目標 Threads 使用者帳號 (預設: make_investment_easy)")
    parser.add_argument("--input", default="threads_posts.csv", help="輸入 CSV 檔案路徑")
    parser.add_argument("--output", default="combined_threads_posts.csv", help="輸出 CSV 檔案路徑")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(script_dir, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    if not os.path.exists(input_path):
        print(f"[!] 錯誤：找不到輸入檔案 '{input_path}'", flush=True)
    else:
        try:
            process_csv_threads(input_path, output_path)
        except Exception as e:
            print(f"[!] 處理過程中發生錯誤: {e}", flush=True)
