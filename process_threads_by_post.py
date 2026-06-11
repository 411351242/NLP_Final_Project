import os
import re
import pandas as pd
import argparse

def process_csv_threads(csv_path, output_path):
    print(f"正在讀取 {csv_path}...")
    # 支援不同編碼的讀取
    for encoding in ['utf-8-sig', 'utf-8', 'cp950']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"成功使用 {encoding} 編碼讀取 CSV。")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"無法讀取檔案 {csv_path}，請檢查檔案編碼是否為 utf-8 或 cp950。")

    # 檢查必要的欄位是否存在
    required_cols = {'貼文與串文編號', '文字內容'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 檔案缺少必要欄位：{required_cols - set(df.columns)}")

    # 解析貼文與串文編號，以便正確進行數值排序
    def parse_id(id_str):
        if not isinstance(id_str, str):
            return 0, 0
        match = re.match(r'^貼文(\d+)_第(\d+)串$', id_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    parsed = df['貼文與串文編號'].apply(parse_id)
    df['貼文數字'] = [p[0] for p in parsed]
    df['串順序'] = [p[1] for p in parsed]

    # 依貼文編號與串順序進行數值排序，確保合併時順序正確
    df = df.sort_values(by=['貼文數字', '串順序'])

    # 清除文字內容結尾的「(續」或「續」字樣
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # 匹配並清除結尾的 (續、（續、續等標記
        text_cleaned = re.sub(r'[\(（\s]*續[\s\)*）]*$', '', text)
        return text_cleaned.strip()

    df['文字內容_clean'] = df['文字內容'].apply(clean_text)

    # 重新組裝為 貼文{X} 格式
    df['貼文編號'] = df['貼文數字'].apply(lambda x: f"貼文{x}")

    # 合併同一貼文的所有串文，並以換行連接
    print("正在合併串文...")
    combined = df.groupby('貼文編號', sort=False)['文字內容_clean'].apply(lambda x: '\n'.join(x)).reset_index()
    combined.rename(columns={'文字內容_clean': '文字內容'}, inplace=True)

    # 匯出合併後的資料
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"合併完成，共 {len(combined)} 篇貼文。已存檔至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併 threads_posts.csv 中的串文為單一貼文")
    parser.add_argument("--input", default="threads_posts.csv", help="輸入 CSV 檔案路徑")
    parser.add_argument("--output", default="combined_threads_posts.csv", help="輸出 CSV 檔案路徑")
    
    args = parser.parse_args()
    
    # 取得相對於腳本的絕對路徑，若檔案在當前目錄下不存在則尋找指令稿所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = args.input
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        potential_path = os.path.join(script_dir, input_path)
        if os.path.exists(potential_path):
            input_path = potential_path

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)

    if not os.path.exists(input_path):
        print(f"錯誤：找不到輸入檔案 '{input_path}'")
    else:
        try:
            process_csv_threads(input_path, output_path)
        except Exception as e:
            print(f"處理過程中發生錯誤: {e}")
