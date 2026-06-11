import os
import re
import pandas as pd

def merge_csv_files():
    directory = r"c:\Users\Kevin\Desktop\NLP"
    output_filename = os.path.join(directory, "threads_posts.csv")
    
    # Find all threads_posts-XX.csv files
    pattern = re.compile(r"^threads_posts-(\d+)\.csv$")
    files = []
    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            files.append((int(match.group(1)), f))
            
    # Sort files by index
    files.sort(key=lambda x: x[0])
    
    if not files:
        print("No matching threads_posts-XX.csv files found!")
        return
        
    print("Found files to merge in order:")
    for idx, name in files:
        print(f"  {name}")
        
    # Detect proper encoding by checking utf-8-sig, utf-8, then cp950
    encoding = 'utf-8-sig'
    try:
        with open(os.path.join(directory, files[0][1]), 'r', encoding=encoding) as f:
            f.read(1000)
    except UnicodeDecodeError:
        encoding = 'utf-8'
        try:
            with open(os.path.join(directory, files[0][1]), 'r', encoding=encoding) as f:
                f.read(1000)
        except UnicodeDecodeError:
            encoding = 'cp950'
            
    print(f"Using input encoding: {encoding}")
    
    # Read all DataFrames
    dfs = []
    for idx, filename in files:
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath, encoding=encoding)
        print(f"  Read {len(df)} rows from {filename}")
        dfs.append(df)
        
    # Concat all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    initial_len = len(merged_df)
    
    # Clean up whitespace in text to ensure accurate duplicate detection
    merged_df['文字內容_clean'] = merged_df['文字內容'].astype(str).str.strip()
    
    # Perform deduplication based on "文字內容" (keeping the first occurrence)
    # This acts as a unique join/union on the content column without adding columns.
    duplicate_mask = merged_df.duplicated(subset=['文字內容_clean'], keep='first')
    num_duplicates = duplicate_mask.sum()
    
    merged_df = merged_df[~duplicate_mask].copy()
    merged_df.drop(columns=['文字內容_clean'], inplace=True)
    final_len = len(merged_df)
    
    print(f"\nTotal rows after concatenation: {initial_len}")
    print(f"Duplicates removed based on '文字內容': {num_duplicates}")
    print(f"Final unique rows: {final_len}")
    
    # Save the deduplicated dataframe
    merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Successfully merged and saved unique rows into: {output_filename}")

if __name__ == "__main__":
    merge_csv_files()
