import subprocess
import os
import sys
import pickle
import json
from datetime import datetime
import pandas as pd
from rag_engine import TextCleaner, VectorIndexer

def run_script(script_name):
    print(f"=== Executing {script_name} ===")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        raise RuntimeError(f"Script {script_name} failed.")
    else:
        print(result.stdout)

def main():
    # 1. Run the data harvesting sub-steps
    run_script("custom_threads_scraper.py")
    run_script("step2_extract_posts.py")
    run_script("merge_posts.py")
    
    # 2. Load the freshly merged posts and chunked threads
    posts_file = "combined_threads_posts.csv"
    threads_file = "threads_posts.csv"
    
    if not os.path.exists(posts_file) or not os.path.exists(threads_file):
        raise FileNotFoundError("Merged posts or chunked threads files were not generated.")
        
    df_posts = pd.read_csv(posts_file)
    df_threads = pd.read_csv(threads_file)
    
    print(f"Loaded {len(df_posts)} combined posts and {len(df_threads)} raw threads.")
    
    # 3. Clean and filter language using TextCleaner
    # Clean texts
    df_posts = df_posts.dropna(subset=['文字內容']).copy()
    df_posts['cleaned_text'] = df_posts['文字內容'].apply(TextCleaner.clean_text)
    
    df_threads = df_threads.dropna(subset=['文字內容']).copy()
    df_threads['cleaned_text'] = df_threads['文字內容'].apply(TextCleaner.clean_text)
    
    # Filter language (Keep only Chinese threads/posts)
    print("Filtering language (langdetect)...")
    df_posts_filtered = TextCleaner.filter_by_language(df_posts, 'cleaned_text')
    df_threads_filtered = TextCleaner.filter_by_language(df_threads, 'cleaned_text')
    
    print(f"Language filtering finished:")
    print(f"- Combined posts: {len(df_posts)} -> {len(df_posts_filtered)}")
    print(f"- Chunked threads: {len(df_threads)} -> {len(df_threads_filtered)}")
    
    # 4. Fit the VectorIndexer on combined posts
    print("Re-indexing vector embeddings...")
    indexer = VectorIndexer()
    indexer.fit(df_posts_filtered['cleaned_text'], post_ids=df_posts_filtered['貼文編號'])
    
    # 5. Serialize the VectorIndexer to disk
    indexer_pkl = "embeddings_index.pkl"
    with open(indexer_pkl, "wb") as f:
        pickle.dump(indexer, f)
    print(f"Serialized VectorIndexer to {indexer_pkl}")
    
    # 6. Save metadata for Streamlit sidebar
    metadata = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_posts": len(df_posts_filtered),
        "total_threads": len(df_threads_filtered)
    }
    
    metadata_json = "pipeline_metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"Saved metadata to {metadata_json}")
    print("=== Pipeline completed successfully! ===")

if __name__ == "__main__":
    main()
