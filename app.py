import streamlit as st
import os
import pickle
import json
import subprocess
import sys
import textwrap
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
    
    /* Premium Architecture Card & Elements */
    .arch-card {
        background: linear-gradient(135deg, #1e222b, #151821);
        border: 1px solid #3e4451;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }
    
    .arch-header {
        color: #61afef;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid #2c313c;
        padding-bottom: 10px;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        margin-bottom: 10px;
        background-color: #161a22;
        border-radius: 8px;
        border-left: 5px solid #4b5263;
        transition: all 0.2s ease;
    }
    
    .file-item:hover {
        transform: translateX(6px);
        background-color: #212631;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Different left borders for different file types */
    .file-core { border-left-color: #61afef; }
    .file-pipeline { border-left-color: #98c379; }
    .file-ui { border-left-color: #c678dd; }
    .file-data { border-left-color: #e5c07b; }
    .file-doc { border-left-color: #56b6c2; }
    
    .file-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .file-icon {
        font-size: 1.25rem;
    }
    
    .file-name {
        font-family: 'Fira Code', Consolas, Monaco, monospace;
        font-weight: 600;
        color: #abb2bf;
    }
    
    .badge {
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .badge-core { background-color: rgba(97, 175, 239, 0.15); color: #61afef; border: 1px solid rgba(97, 175, 239, 0.4); }
    .badge-pipeline { background-color: rgba(152, 195, 121, 0.15); color: #98c379; border: 1px solid rgba(152, 195, 121, 0.4); }
    .badge-ui { background-color: rgba(198, 120, 221, 0.15); color: #c678dd; border: 1px solid rgba(198, 120, 221, 0.4); }
    .badge-data { background-color: rgba(229, 192, 123, 0.15); color: #e5c07b; border: 1px solid rgba(229, 192, 123, 0.4); }
    .badge-doc { background-color: rgba(86, 182, 194, 0.15); color: #56b6c2; border: 1px solid rgba(86, 182, 194, 0.4); }
    
    .file-desc {
        color: #828997;
        font-size: 0.9rem;
        max-width: 60%;
        text-align: right;
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

# Sidebar Configuration

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
with st.expander("🗺️ 專案系統架構與 RAG 雙階段資料流 (點擊展開/收合)", expanded=True):
    st.info("💡 **專案系統架構與代碼組織**：以下展示了本專案整個目錄結構、資料管線流程，以及 RAG 雙階段檢索的底層運作原理。")
    
    # Create beautiful tabs
    tab_struct, tab_pipe, tab_rag = st.tabs([
        "📁 專案檔案結構 (Project Structure)", 
        "🔄 資料管線流程 (Data Pipeline Flow)", 
        "🧠 RAG 雙階段檢索 (Dual-Stage RAG)"
    ])
    
    with tab_struct:
        # We will render the beautiful card with file categories
        st.markdown(textwrap.dedent("""
        <div class="arch-card">
            <div class="arch-header">
                <span>📁 FinalProject 專案完整目錄結構</span>
            </div>
            <p style="color: #abb2bf; font-size: 0.95rem; margin-bottom: 20px;">
                本專案採用模組化設計，清晰劃分了前端 UI、RAG 核心引擎、自動化資料管線以及知識庫儲存介面。以下為各模組檔案說明：
            </p>
            
            <!-- Category: UI Interface -->
            <div style="margin-top: 15px; margin-bottom: 5px; color: #c678dd; font-weight: bold; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                🖥️ 使用者介面 (Web UI)
            </div>
            <div class="file-item file-ui">
                <div class="file-left">
                    <span class="file-icon">📄</span>
                    <span class="file-name">app.py</span>
                    <span class="badge badge-ui">前端介面</span>
                </div>
                <div class="file-desc">Streamlit 互動網頁主程式，負責對話狀態、UI 渲染與參數控制</div>
            </div>
            
            <!-- Category: Core Engine -->
            <div style="margin-top: 20px; margin-bottom: 5px; color: #61afef; font-weight: bold; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                ⚙️ 核心邏輯 (Core Engine)
            </div>
            <div class="file-item file-core">
                <div class="file-left">
                    <span class="file-icon">📄</span>
                    <span class="file-name">rag_engine.py</span>
                    <span class="badge badge-core">核心引擎</span>
                </div>
                <div class="file-desc">RAG 核心運算引擎 (文字清洗、向量檢索、精篩重排序與慢速頻率防護)</div>
            </div>
            
            <!-- Category: Data Pipeline -->
            <div style="margin-top: 20px; margin-bottom: 5px; color: #98c379; font-weight: bold; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                🔄 資料管線 (Data Pipeline)
            </div>
            <div class="file-item file-pipeline">
                <div class="file-left">
                    <span class="file-icon">📄</span>
                    <span class="file-name">update_pipeline.py</span>
                    <span class="badge badge-pipeline">管線協調</span>
                </div>
                <div class="file-desc">自動化更新協調器，一鍵調用爬蟲、重組與向量索引重構</div>
            </div>
            <div class="file-item file-pipeline">
                <div class="file-left">
                    <span class="file-icon">📄</span>
                    <span class="file-name">custom_threads_scraper.py</span>
                    <span class="badge badge-pipeline">貼文爬蟲</span>
                </div>
                <div class="file-desc">Threads 貼文模擬爬取腳本，用於模擬獲取最新社群數據</div>
            </div>
            <div class="file-item file-pipeline">
                <div class="file-left">
                    <span class="file-icon">📄</span>
                    <span class="file-name">step2_extract_posts.py</span>
                    <span class="badge badge-pipeline">資料增量</span>
                </div>
                <div class="file-desc">貼文資料去重與增量追加 (raw_scraped_posts.csv -> threads_posts.csv)</div>
            </div>
            <div class="file-item file-pipeline">
                <div class="file-left">
                    <span class="file-icon">📄</span>
                    <span class="file-name">merge_posts.py</span>
                    <span class="badge badge-pipeline">串文重組</span>
                </div>
                <div class="file-desc">串文按 Post ID 排序與重組，將單篇發言串連為對話鏈結構</div>
            </div>
            
            <!-- Category: Databases -->
            <div style="margin-top: 20px; margin-bottom: 5px; color: #e5c07b; font-weight: bold; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                📊 資料儲存與索引 (Database & Storage)
            </div>
            <div class="file-item file-data">
                <div class="file-left">
                    <span class="file-icon">💾</span>
                    <span class="file-name">embeddings_index.pkl</span>
                    <span class="badge badge-data">向量索引</span>
                </div>
                <div class="file-desc">輕量化向量序列化檔，排除 PyTorch 模型權重，降低部署空間 (僅 1.81 MB)</div>
            </div>
            <div class="file-item file-data">
                <div class="file-left">
                    <span class="file-icon">📊</span>
                    <span class="file-name">combined_threads_posts.csv</span>
                    <span class="badge badge-data">合併貼文集</span>
                </div>
                <div class="file-desc">最終經資料清洗與對話鏈整併後，用於 RAG 核心檢索的資料庫</div>
            </div>
            <div class="file-item file-data">
                <div class="file-left">
                    <span class="file-icon">📊</span>
                    <span class="file-name">threads_posts.csv</span>
                    <span class="badge badge-data">原始貼文集</span>
                </div>
                <div class="file-desc">經增量爬蟲寫入、去重處理後的原始單篇 Threads 貼文集</div>
            </div>
            <div class="file-item file-data">
                <div class="file-left">
                    <span class="file-icon">⚙️</span>
                    <span class="file-name">pipeline_metadata.json</span>
                    <span class="badge badge-data">中繼數據</span>
                </div>
                <div class="file-desc">儲存管線執行狀態、最後更新時間與貼文統計數據的 JSON 檔案</div>
            </div>
            
            <!-- Category: Documentation -->
            <div style="margin-top: 20px; margin-bottom: 5px; color: #56b6c2; font-weight: bold; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                📝 專案文件與配置 (Docs & Configs)
            </div>
            <div class="file-item file-doc">
                <div class="file-left">
                    <span class="file-icon">📝</span>
                    <span class="file-name">preprocessing_recommendations.md</span>
                    <span class="badge badge-doc">前處理建議</span>
                </div>
                <div class="file-desc">詳細規劃資料清洗規則、繁簡轉換與語言過濾的指導方針</div>
            </div>
            <div class="file-item file-doc">
                <div class="file-left">
                    <span class="file-icon">📝</span>
                    <span class="file-name">PRD.md</span>
                    <span class="badge badge-doc">需求文件</span>
                </div>
                <div class="file-desc">專案核心功能與技術規格的產品需求文件</div>
            </div>
            <div class="file-item file-doc">
                <div class="file-left">
                    <span class="file-icon">📋</span>
                    <span class="file-name">requirements.txt</span>
                    <span class="badge badge-doc">依賴套件</span>
                </div>
                <div class="file-desc">專案於 Streamlit Cloud 部署時所需之 Python 套件依賴清單</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
    with tab_pipe:
        st.markdown("### 🔄 資料管線流程圖 (Data Pipeline Flowchart)")
        st.markdown("以下為專案的自動化更新管線運作流程，說明新爬取的 Threads 貼文如何一步步被清洗、對話重組並存入向量索引：")
        
        st.markdown(textwrap.dedent("""
        ```mermaid
        graph TD
            %% Define styles
            classDef pipeline fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#ecf0f1;
            classDef data fill:#d35400,stroke:#e67e22,stroke-width:2px,color:#fff;
            classDef coord fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff;

            subgraph Data Pipeline (資料處理管線)
                A[custom_threads_scraper.py] -->|1. 模擬爬蟲抓取| B[(raw_scraped_posts.csv)]
                B -->|2. 增量追加與去重| C[step2_extract_posts.py]
                C -->|3. 更新主資料庫| D[(threads_posts.csv)]
                D -->|4. 按 Post ID 排序與對話重組| E[merge_posts.py]
                E -->|5. 產出完整對話鏈結構| F[(combined_threads_posts.csv)]
            end

            subgraph Re-indexing Pipeline (向量重索引)
                F -->|6. 載入並執行中文過濾與清洗| G[update_pipeline.py]
                G -->|7. 輕量 SentenceTransformer 編碼| H[(embeddings_index.pkl)]
                G -->|8. 寫入執行中繼資料| I[(pipeline_metadata.json)]
            end

            class A,C,E pipeline;
            class B,D,F,H,I data;
            class G coord;
        ```
        """))
        
        st.markdown(textwrap.dedent("""
        #### ⚙️ 更新管線階段說明：
        1. **增量資料獲取**：`custom_threads_scraper.py` 模擬向 Threads API/網頁端拉取最新發表的貼文數據。
        2. **去重與追加**：`step2_extract_posts.py` 比對現有 `threads_posts.csv` 中的發文 ID，只將全新發布的貼文增量追加進去，防止重複檢索。
        3. **對話鏈重組**：`merge_posts.py` 根據 `thread_id` 將多篇串文拼合在一起，形成完整的上下文（Context），避免語意切碎。
        4. **自動化清洗與向量化**：`update_pipeline.py` 一鍵調用上述流程，隨後讀取數據，過濾無效英文或亂碼，呼叫 Embedding 模型重算特徵並儲存至輕量索引中。
        """))

    with tab_rag:
        st.markdown("### 🧠 雙階段高精準檢索與安全生成流程 (Dual-Stage Retrieval)")
        st.markdown("以下為系統在接收到使用者輸入提問後的 RAG 核心處理流程：")
        
        st.markdown(textwrap.dedent("""
        ```mermaid
        graph TD
            classDef query fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff;
            classDef engine fill:#16a085,stroke:#1abc9c,stroke-width:2px,color:#fff;
            classDef security fill:#c0392b,stroke:#e74c3c,stroke-width:2px,color:#fff;
            classDef model fill:#d35400,stroke:#e67e22,stroke-width:2px,color:#fff;

            A[使用者提問] -->|1. 傳送問題| B(TextCleaner)
            B -->|2. 文本清洗與簡繁轉換| C[VectorIndexer]
            C -->|3. 載入 embeddings_index.pkl| D[(輕量向量資料庫)]
            C -->|4. 第一階段: Cosine Similarity 檢索 Top-10| E[CrossEncoderReranker]
            E -->|5. 第二階段: 交叉比對相關性重排序| F[選出 Top-K 候選貼文]
            F -->|6. 合併 Context (限 3000 字)| G[GeminiGenerator]
            H[RateLimiter] -->|7. 安全防護 RPM<=10 / RPD<=200| G
            G -->|8. 生成回答| I[Streamlit 聊天對話框]

            class A,I query;
            class B,C,E,F engine;
            class H security;
            class G,D model;
        ```
        """))
        
        st.markdown(textwrap.dedent("""
        #### 🔍 檢索與生成優化機制：
        1. **第一階段（粗篩/召回）**: 
           * 使用 `paraphrase-multilingual-MiniLM-L12-v2` 輕量多語言模型，將使用者問題向量化，在本地 `embeddings_index.pkl` 中快速計算餘弦相似度，高效率地篩選出 **Top-10** 最相關的候選貼文。
        2. **第二階段（精篩/重排序）**:
           * 導入 `Cross-Encoder (ms-marco-MiniLM-L-6-v2)`。Cross-Encoder 會將使用者問題與 Top-10 候選文檔一對一進行交叉比對，計算更精細的深度相關性分數，有效防範單純向量檢索被局部關鍵字誤導的問題。
           * 最終只取評分最高的 **Top-K（預設 3）** 片段送往 Gemini 生成回答。
        3. **安全防範與頻率防護 (Rate Limiting & Safeguards)**:
           * **動態長度安全閥**：限制 Context 總長不超過 3000 字（約 2000 tokens），保障生成延遲（Latency）小於 8 秒。
           * **慢速限制模式（Slow Mode）**：內建 **RPM <= 10（每分鐘最高 10 次）** 與 **RPD <= 200（每日最高 200 次）** 安全保護，若發言頻率過高，系統將自動延遲等待，以防展示期間 API 金鑰被刷爆或停權。
        """))

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
