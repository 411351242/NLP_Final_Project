import pandas as pd
import os

def extract_and_append():
    raw_file = "raw_scraped_posts.csv"
    main_file = "threads_posts.csv"
    
    if not os.path.exists(raw_file):
        print(f"Error: {raw_file} does not exist.")
        return
        
    df_raw = pd.read_csv(raw_file)
    
    if os.path.exists(main_file):
        df_main = pd.read_csv(main_file)
        print(f"Loaded existing {main_file} with {len(df_main)} records.")
    else:
        df_main = pd.DataFrame(columns=["貼文與串文編號", "文字內容"])
        print(f"Created new {main_file}.")
        
    # Combine and drop duplicates based on '貼文與串文編號'
    df_combined = pd.concat([df_main, df_raw], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["貼文與串文編號"], keep="last")
    
    df_combined.to_csv(main_file, index=False, encoding="utf-8")
    print(f"Appended records. Total rows in {main_file} now: {len(df_combined)}")

if __name__ == "__main__":
    extract_and_append()
