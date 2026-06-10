import streamlit as st
import os
import pickle
import json
import subprocess
import sys
from dotenv import load_dotenv
from rag_engine import TextCleaner, VectorIndexer, CrossEncoderReranker, GeminiGenerator

# Set Streamlit page style and layout
st.set_page_config(
    page_title="Threads 財經與職涯智慧 RAG 問答助理 v2.0",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #0f1115;
    }
    .stChatInput {
        border-radius: 20px;
    }
    .st-emotion-cache-1c7n2qd {
        background-color: #1b1e23;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache the heavy models
@st.cache_resource
def load_indexer(index_path):
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_reranker():
    return CrossEncoderReranker()

# Index file path
INDEX_PATH = "embeddings_index.pkl"
METADATA_PATH = "pipeline_metadata.json"

# Load metadata
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": "尚未更新", "total_posts": 0, "total_threads": 0}

metadata = load_metadata()

# ================= SIDEBAR =================
st.sidebar.title("💼 智慧庫配置與狀態")

# Toggle architecture diagram
if "show_architecture" not in st.session_state:
    st.session_state.show_architecture = False

if st.sidebar.button("🗺️ 查看系統架構與 RAG 流程圖", use_container_width=True):
    st.session_state.show_architecture = not st.session_state.show_architecture

st.sidebar.subheader("系統狀態")
st.sidebar.markdown(f"**📚 總貼文數（已過濾語系）**: `{metadata.get('total_posts', 0)}` 篇")
st.sidebar.markdown(f"**💬 總串文數（原始拆分）**: `{metadata.get('total_threads', 0)}` 條")
st.sidebar.markdown(f"**🕒 知識庫更新時間**: `{metadata.get('last_updated', '未知')}`")

st.sidebar.divider()

# Trigger Pipeline Refresh (with Admin Password protection to prevent public access of scraper)
st.sidebar.subheader("知識庫更新")
admin_password = st.sidebar.text_input("🔒 管理員密碼 (更新語料庫)", type="password", help="為防範爬蟲憑證或 Cookie 洩漏，僅限管理員輸入密碼後方能手動觸發資料更新。")

# Load password from environment variable (.env) or Streamlit Secrets safely
expected_password = os.getenv("ADMIN_PASSWORD")
if not expected_password:
    try:
        expected_password = st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        expected_password = None

if expected_password and admin_password == expected_password:
    st.sidebar.success("管理員身份驗證成功！")
    if st.sidebar.button("🔄 一鍵爬取並更新知識庫", use_container_width=True):
        with st.spinner("正在執行更新管線 (update_pipeline)... 包含爬取、清洗與重建索引..."):
            try:
                # Execute pipeline script using sys.executable to ensure virtualenv context
                result = subprocess.run([sys.executable, "update_pipeline.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.sidebar.success("更新成功！")
                    st.cache_resource.clear()  # Clear Streamlit cache to load new indexer
                    st.rerun()
                else:
                    st.sidebar.error(f"更新失敗！\nError: {result.stderr}")
            except Exception as e:
                st.sidebar.error(f"執行出錯: {str(e)}")
elif admin_password:
    st.sidebar.error("密碼錯誤，無法解鎖更新功能。")
else:
    st.sidebar.info("請輸入管理員密碼以啟用更新功能。")

st.sidebar.divider()

# Hyperparameter Controls
st.sidebar.subheader("檢索與生成參數")
temperature = st.sidebar.slider("LLM 溫度 (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                                help="控制生成答案的創造性與保守度。值越低回答越嚴謹、貼合原文。")
top_k_rerank = st.sidebar.slider("精篩重排序選取數 (Top-K Rerank)", min_value=1, max_value=5, value=3, step=1,
                                 help="第二階段 Cross-Encoder 從候選貼文中重排序選取最相關的前幾篇送給 Gemini。")

st.sidebar.divider()

# API Key Check and fallback
st.sidebar.subheader("API 配置")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key_input = st.sidebar.text_input("請輸入 Gemini API Key:", type="password")
    if api_key_input:
        api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.sidebar.warning("請於此處或環境變數中設定 GOOGLE_API_KEY 才能正常對話。")

# ================= MAIN PAGE =================
st.title("💼 Threads 財經與職涯智慧 RAG 問答助理 v2.0")
st.markdown("""
本問答助理採用**雙階段高精準檢索架構 (Dense + Re-ranking)**：
1. **第一階段 (初篩/召回)**：使用 `paraphrase-multilingual-MiniLM-L12-v2` 檢索 Top-10 候選貼文。
2. **第二階段 (精篩/重排序)**：使用 `CrossEncoder` 模型進行一對一交叉比對評分，精篩最相關的貼文送至 Gemini。
3. **LLM 生成**：由 `Gemini 3.1` 嚴格遵循參考上下文，以原作者（前投行交易員）風格產出專業繁中分析。
""")

# Render system architecture if toggled
if st.session_state.show_architecture:
    st.info("💡 **系統架構與資料流**：您可以在此查看本專案從資料爬取、清洗、向量召回到重排序生成回答的完整生命週期流程。")
    st.markdown("""
### 🗺️ Threads RAG 智慧助理 v2.0 系統架構圖

```text
                     【原始資料集】
                       /        \\
                      /          \\
    [流程 A: 合併貼文集]          [流程 B: 原始單篇串文集]
    (combined_threads_posts.csv)  (threads_posts.csv)
              |                             |
              ▼                             ▼
     [資料清洗管道: 刪除換行、過濾 unicode 亂碼 (如 ￼) 與平台尾綴雜訊]
              |                             |
              ▼                             ▼
     [流程 A 向量索引建立]         [流程 B 向量索引建立]
     (indexer_merged)              (indexer_chunked)
              \\                             /
               \\                           /
              【使用者提問】: 提供相同問題進行雙軌檢索
                 |                       |
                 ▼                       ▼
              RAG 檢索(Top-2)         RAG 檢索(Top-4)
                 |                       |
                 ▼                       ▼
              Gemini 生成             Gemini 生成
                 \\                       /
                  \\                     /
                  【雙流程側邊對比與性能分析】
```

#### 🔄 雙階段高精準檢索架構 (Two-stage Retrieval Pipeline)
1. **第一階段（粗篩 - 召回）**: 
   * 使用 `paraphrase-multilingual-MiniLM-L12-v2` 輕量多語言模型，將使用者問題向量化，在本地 `embeddings_index.pkl` 中計算餘弦相似度，篩選出 **Top-10** 最相關的候選貼文。
2. **第二階段（精篩 - 重排序）**:
   * 導入 `Cross-Encoder (ms-marco-MiniLM-L-6-v2)`。Cross-Encoder 會將使用者問題與 Top-10 候選文檔一對一交叉比對，計算更細緻的相關性分數，有效防範單純向量檢索被局部詞彙誤導。
   * 最終只取評分最高的 **Top-K（預設 3）** 片段送往 Gemini 生成回答。
3. **API 安全防護與慢速模式 (Rate Limiting & Safeguards)**:
   * **動態長度限制**：字數安全閥限制 Context 總長不超過 3000 字（約 2000 tokens），保障 Latency 小於 8 秒。
   * **慢速限制模式（Slow Mode）**：內建 **RPM <= 10（每分鐘最高 10 次）** 與 **RPD <= 200（每日最高 200 次）** 安全保護。若發言頻率過高，系統將自動延遲等待，以防展示期間 API 金鑰耗盡。
---
""")

# Load indexer and reranker
indexer = load_indexer(INDEX_PATH)
reranker = load_reranker()

if not indexer:
    st.error("找不到向量索引檔 `embeddings_index.pkl`！請先點擊側邊欄的「一鍵爬取並更新知識庫」來初始化索引。")
else:
    # Initialize Gemini Generator
    generator = None
    if api_key:
        generator = GeminiGenerator(api_key=api_key)

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render sources if they exist for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("🔍 查看參考來源與相關性評分"):
                    for i, src in enumerate(msg["sources"]):
                        score_type = 'rerank_score' if 'rerank_score' in src else 'score'
                        st.markdown(f"**[{i+1}] 貼文編號: `{src['post_id']}` (重排序分數: `{src[score_type]:.4f}`)**")
                        st.text_area(f"貼文內容 {i+1}", value=src['document'], height=120, disabled=True, label_visibility="collapsed")

    # Accept User Input
    if user_query := st.chat_input("請輸入您的財經或職涯問題..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        if not generator:
            with st.chat_message("assistant"):
                st.error("未設定有效的 API Key，無法生成回答。請在側邊欄填寫 API Key。")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("正在進行兩階段語意檢索與精篩重排序..."):
                    # Step 1: Dense Retrieval (Recall Top-10)
                    top_10_candidates = indexer.search(user_query, top_k=10)
                    
                    # Step 2: Cross-Encoder Re-ranking (Select Top-K)
                    final_retrieved = reranker.rerank(user_query, top_10_candidates, top_k=top_k_rerank)
                    
                    # Step 3: Dynamic Token/Character limit protection
                    final_retrieved_docs = []
                    # Keep track of indices we include to get their source metadata
                    current_chars = 0
                    for res in final_retrieved:
                        # Character token safeguard limit
                        if current_chars + len(res['document']) > 3000:
                            break
                        final_retrieved_docs.append(res)
                        current_chars += len(res['document'])

                with st.spinner("正在傳送至 Gemini 生成回答..."):
                    try:
                        response_text = generator.generate(
                            query=user_query,
                            retrieved_docs=final_retrieved_docs,
                            temperature=temperature
                        )
                        message_placeholder.markdown(response_text)
                        
                        # Save assistant message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": final_retrieved_docs
                        })
                        
                        # Render retrieved sources expander
                        with st.expander("🔍 查看參考來源與相關性評分"):
                            for i, src in enumerate(final_retrieved_docs):
                                st.markdown(f"**[{i+1}] 貼文編號: `{src['post_id']}` (重排序分數: `{src['rerank_score']:.4f}`)**")
                                st.text_area(f"貼文內容 {i+1}", value=src['document'], height=120, disabled=True, label_visibility="collapsed")
                                
                    except Exception as e:
                        st.error(f"對話生成出錯: {str(e)}")
