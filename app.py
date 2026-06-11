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
    .reportview-container { background-color: #0f1115; }
    .stChatInput { border-radius: 20px; }
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
st.sidebar.markdown(f"**📚 總貼文數（知識庫）**: `{metadata.get('total_posts', 0)}` 篇")
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
                result = subprocess.run([sys.executable, "update_pipeline.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.sidebar.success("更新成功！")
                    st.cache_resource.clear()
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
3. **LLM 生成**：由 `Gemini` 嚴格遵循參考上下文，以原作者（前投行交易員）風格產出專業繁中分析。
""")

# ================= ARCHITECTURE PANEL =================
# Helper: build a single file-row HTML string (no indentation issues)
def _file_row(icon, name, badge, badge_color, border_color, desc):
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'padding:10px 15px;margin-bottom:8px;background:#161a22;'
        f'border-radius:8px;border-left:4px solid {border_color};">'
        '<div style="display:flex;align-items:center;gap:10px;flex:1;">'
        f'<span style="font-size:1.1rem;">{icon}</span>'
        f'<code style="font-size:0.88rem;color:#abb2bf;font-weight:600;">{name}</code>'
        f'<span style="padding:2px 9px;border-radius:20px;font-size:0.7rem;font-weight:700;'
        f'background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}88;">'
        f'{badge}</span>'
        '</div>'
        f'<div style="color:#636d83;font-size:0.8rem;text-align:right;max-width:52%;">{desc}</div>'
        '</div>'
    )

def _section_header(icon, title, color):
    return (
        f'<div style="margin-top:18px;margin-bottom:6px;color:{color};'
        f'font-weight:700;font-size:0.88rem;letter-spacing:0.3px;">{icon} {title}</div>'
    )

def _flow_step(num, color, icon, title, subtitle):
    return (
        '<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:4px;">'
        f'<div style="flex-shrink:0;width:42px;height:42px;border-radius:50%;'
        f'background:{color}22;border:2px solid {color};display:flex;align-items:center;'
        f'justify-content:center;font-size:1.1rem;">{icon}</div>'
        '<div style="padding-top:2px;">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-size:0.7rem;color:{color};font-weight:700;">{num}</span>'
        f'<span style="font-weight:700;color:#abb2bf;font-size:0.92rem;">{title}</span>'
        '</div>'
        f'<div style="color:#636d83;font-size:0.8rem;">{subtitle}</div>'
        '</div></div>'
    )

def _connector(color):
    return f'<div style="width:2px;height:14px;background:{color}55;margin:0 auto;margin-left:20px;"></div>'

def _pipe_box(color, icon, name, sub):
    return (
        f'<div style="background:{color}18;border:1.5px solid {color}66;border-radius:8px;'
        f'padding:8px 12px;text-align:center;min-width:110px;">'
        f'<div style="font-size:1.1rem;">{icon}</div>'
        f'<div style="color:{color};font-weight:700;font-size:0.8rem;">{name}</div>'
        f'<div style="color:#636d83;font-size:0.68rem;">{sub}</div>'
        '</div>'
    )

def _arrow(label=""):
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;color:#4b5263;font-size:1rem;padding:0 6px;flex-shrink:0;">'
        f'→<span style="font-size:0.62rem;color:#555;">{label}</span></div>'
    )

def _pipe_row(*items):
    return (
        '<div style="display:flex;align-items:center;gap:2px;margin-bottom:12px;flex-wrap:wrap;">'
        + "".join(items) + '</div>'
    )

# Render system architecture if toggled
if st.session_state.show_architecture:
    st.info("💡 **專案系統架構與代碼組織**：以下展示了本專案整個目錄結構、資料管線流程，以及 RAG 雙階段檢索的底層運作原理。")

    tab_struct, tab_pipe, tab_rag = st.tabs([
        "📁 專案檔案結構",
        "🔄 資料管線流程",
        "🧠 RAG 雙階段檢索"
    ])

    # ── TAB 1: Directory Structure ────────────────────────────────────────────
    with tab_struct:
        st.markdown("#### 📁 FinalProject 專案完整目錄結構")
        st.caption("本專案採用模組化設計，清晰劃分了前端 UI、RAG 核心引擎、自動化資料管線以及知識庫儲存介面。")

        html = (
            '<div style="background:linear-gradient(135deg,#1e222b,#151821);'
            'border:1px solid #3e4451;border-radius:12px;padding:20px;'
            'box-shadow:0 8px 32px rgba(0,0,0,.35);">'
        )
        html += _section_header("🖥️", "使用者介面 (Web UI)", "#c678dd")
        html += _file_row("📄","app.py","前端介面","#c678dd","#c678dd","Streamlit 互動網頁主程式，負責對話狀態、UI 渲染與參數控制")

        html += _section_header("⚙️", "核心邏輯 (Core Engine)", "#61afef")
        html += _file_row("📄","rag_engine.py","核心引擎","#61afef","#61afef","RAG 核心運算引擎（TextCleaner、VectorIndexer、Reranker、RateLimiter）")

        html += _section_header("🔄", "資料管線 (Data Pipeline)", "#98c379")
        html += _file_row("📄","update_pipeline.py","管線協調","#98c379","#98c379","自動化更新協調器，一鍵調用爬蟲、重組與向量索引重構")
        html += _file_row("📄","custom_threads_scraper.py","貼文爬蟲","#98c379","#98c379","Threads 貼文模擬爬取腳本，用於模擬獲取最新社群數據")
        html += _file_row("📄","step2_extract_posts.py","資料增量","#98c379","#98c379","貼文去重與增量追加（raw_scraped_posts.csv → threads_posts.csv）")
        html += _file_row("📄","merge_posts.py","串文重組","#98c379","#98c379","串文按 Post ID 排序重組，將單篇發言串連為對話鏈結構")

        html += _section_header("📊", "資料儲存與索引 (Storage)", "#e5c07b")
        html += _file_row("💾","embeddings_index.pkl","向量索引","#e5c07b","#e5c07b","輕量化向量序列化檔，排除 PyTorch 模型權重（僅 1.81 MB）")
        html += _file_row("📊","combined_threads_posts.csv","合併貼文集","#e5c07b","#e5c07b","最終對話鏈整併後，供 RAG 核心檢索的知識庫")
        html += _file_row("📊","threads_posts.csv","原始貼文集","#e5c07b","#e5c07b","經增量爬蟲寫入、去重處理後的原始單篇貼文集")
        html += _file_row("⚙️","pipeline_metadata.json","中繼數據","#e5c07b","#e5c07b","儲存管線執行狀態、最後更新時間與貼文統計數據")

        html += _section_header("📝", "專案文件與配置 (Docs)", "#56b6c2")
        html += _file_row("📝","preprocessing_recommendations.md","前處理建議","#56b6c2","#56b6c2","詳細規劃資料清洗規則、繁簡轉換與語言過濾的指導方針")
        html += _file_row("📝","PRD.md","需求文件","#56b6c2","#56b6c2","專案核心功能與技術規格的產品需求文件")
        html += _file_row("📋","requirements.txt","依賴套件","#56b6c2","#56b6c2","Streamlit Cloud 部署時所需的 Python 套件依賴清單")
        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

    # ── TAB 2: Data Pipeline Flow ──────────────────────────────────────────────
    with tab_pipe:
        st.markdown("#### 🔄 資料管線流程圖（Data Pipeline）")
        st.caption("新爬取的 Threads 貼文如何一步步被清洗、對話重組並存入向量索引：")

        pipe_html = (
            '<div style="background:#1a1d25;border:1px solid #3e4451;border-radius:12px;padding:20px;">'
        )
        pipe_html += (
            '<div style="color:#98c379;font-weight:700;margin-bottom:14px;font-size:0.88rem;">'
            '📦 第一階段：資料爬取 → 清洗 → 重組</div>'
        )
        pipe_html += _pipe_row(
            _pipe_box("#98c379","🕷️","custom_threads_scraper",".py"),
            _arrow("爬取"),
            _pipe_box("#e5c07b","📄","raw_scraped_posts",".csv"),
            _arrow("去重追加"),
            _pipe_box("#98c379","🔍","step2_extract_posts",".py"),
            _arrow("寫入"),
            _pipe_box("#e5c07b","📊","threads_posts",".csv"),
        )
        pipe_html += _pipe_row(
            _pipe_box("#98c379","🔗","merge_posts",".py"),
            _arrow("對話鏈重組"),
            _pipe_box("#e5c07b","📊","combined_threads_posts",".csv"),
        )
        pipe_html += (
            '<div style="color:#8e44ad;font-weight:700;margin:16px 0 12px;font-size:0.88rem;'
            'border-top:1px solid #3e4451;padding-top:14px;">'
            '🧮 第二階段：向量化重索引</div>'
        )
        pipe_html += _pipe_row(
            _pipe_box("#8e44ad","⚙️","update_pipeline",".py"),
            _arrow("清洗+編碼"),
            _pipe_box("#e5c07b","💾","embeddings_index",".pkl"),
            _arrow("記錄狀態"),
            _pipe_box("#e5c07b","📋","pipeline_metadata",".json"),
        )
        pipe_html += "</div>"
        st.markdown(pipe_html, unsafe_allow_html=True)

        st.markdown("""
**更新管線四步驟說明：**
1. 🕷️ **增量爬取**：`custom_threads_scraper.py` 模擬拉取最新的 Threads 貼文資料。
2. 🔍 **去重追加**：`step2_extract_posts.py` 比對 Post ID，僅追加全新貼文，防止重複。
3. 🔗 **對話鏈重組**：`merge_posts.py` 將多篇串文按 thread_id 拼合為完整上下文，避免語意切碎。
4. 🧮 **清洗與向量化**：`update_pipeline.py` 過濾非中文文本，呼叫 SentenceTransformer 重算特徵並序列化索引。
        """)

    # ── TAB 3: RAG Dual-Stage Retrieval ───────────────────────────────────────
    with tab_rag:
        st.markdown("#### 🧠 雙階段高精準檢索與安全生成流程（RAG）")
        st.caption("使用者提問後，系統依序經過以下 8 個處理步驟生成回答：")

        rag_steps = [
            ("#2980b9", "🗣️",  "Step 1",  "使用者提問",          "輸入財經或職涯問題"),
            ("#16a085", "🧹",  "Step 2",  "TextCleaner",         "文本清洗、URL 移除、標準化"),
            ("#16a085", "📐",  "Step 3",  "VectorIndexer",       "第一階段：Cosine Similarity 召回 Top-10"),
            ("#e5c07b", "💾",  "Step 4",  "embeddings_index.pkl","載入本地向量資料庫"),
            ("#16a085", "🔬",  "Step 5",  "CrossEncoderReranker","第二階段：交叉比對重排序 Top-K"),
            ("#c0392b", "🛡️",  "Step 6",  "RateLimiter",         "RPM ≤ 10 / RPD ≤ 200 安全限速防護"),
            ("#d35400", "✨",  "Step 7",  "GeminiGenerator",     "LLM 生成回答（繁中 / 嚴格 RAG 限制）"),
            ("#2980b9", "💬",  "Step 8",  "Streamlit 介面",      "顯示回答與可展開的參考來源"),
        ]

        rag_html = '<div style="background:#1a1d25;border:1px solid #3e4451;border-radius:12px;padding:20px;">'
        for i, (color, icon, step_label, title, subtitle) in enumerate(rag_steps):
            rag_html += _flow_step(step_label, color, icon, title, subtitle)
            if i < len(rag_steps) - 1:
                rag_html += _connector(color)
        rag_html += "</div>"
        st.markdown(rag_html, unsafe_allow_html=True)

        st.markdown("""
**三大核心優化機制：**
1. 🎯 **第一階段（粗篩召回）**：`paraphrase-multilingual-MiniLM-L12-v2` 向量模型快速計算 Cosine Similarity，召回 **Top-10** 候選貼文。
2. 🔬 **第二階段（精篩重排序）**：`Cross-Encoder (ms-marco-MiniLM-L-6-v2)` 對問題與每篇候選文檔進行一對一深度比對，精選最相關的 **Top-K（預設 3）** 片段。
3. 🛡️ **API 安全防護（Slow Mode）**：內建 **RPM ≤ 10、RPD ≤ 200** 限速保護；Context 動態限縮至 3000 字以內，確保展示期間 API 金鑰不爆炸。
        """)

    st.divider()

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
                    current_chars = 0
                    for res in final_retrieved:
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
