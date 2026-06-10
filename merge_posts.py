import pandas as pd
import re
import os

def merge_threads():
    input_file = "threads_posts.csv"
    output_file = "combined_threads_posts.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist. Cannot merge.")
        return
        
    df = pd.read_csv(input_file)
    
    col_id = df.columns[0]
    col_text = df.columns[1]
    
    # Extract post_id (e.g. '貼文1') and thread index (e.g. 2 from '串2')
    df['post_id'] = df[col_id].apply(lambda x: str(x).split('_')[0])
    
    def extract_thread_num(x):
        matches = re.findall(r'\d+', str(x).split('_')[1]) if '_' in str(x) else []
        return int(matches[0]) if matches else 0
        
    df['thread_num'] = df[col_id].apply(extract_thread_num)
    
    # Sort threads sequentially
    df = df.sort_values(by=['post_id', 'thread_num'])
    
    # Merge thread texts
    grouped = df.groupby('post_id')[col_text].apply(lambda x: '\n'.join(x.dropna().astype(str))).reset_index()
    grouped.columns = ['貼文編號', '文字內容']
    
    grouped.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Successfully merged {len(df)} threads into {len(grouped)} combined posts in {output_file}.")

if __name__ == "__main__":
    merge_threads()
