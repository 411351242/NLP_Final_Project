import streamlit as st
import os
import pickle
import json
import subprocess
import sys
import textwrap
from dotenv import load_dotenv
from rag_engine import TextCleaner, VectorIndexer, CrossEncoderReranker, GeminiGenerator

# 設定 Streamlit 頁面標題與寬螢幕配置
st.set_page_config(
    page_title="Threads 財經與職涯智慧 RAG 問答助理 v2.0",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂介面 CSS 樣式
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

# 載入環境變數
load_dotenv()

# 初始化對話歷程 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 快取載入大型資源模型，避免每次互動重複載入
@st.cache_resource
def load_indexer(index_path: str):
    """載入向量索引物件"""
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_reranker():
    """載入 Cross-Encoder 重排序模型"""
    return CrossEncoderReranker()

# 索引與中繼資料路徑
INDEX_PATH = "embeddings_index.pkl"
METADATA_PATH = "pipeline_metadata.json"

def load_metadata() -> dict:
    """讀取資料管線中繼統計資料"""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": "尚未更新", "total_posts": 0, "total_threads": 0}

metadata = load_metadata()

# ================= 側邊欄控制項 (SIDEBAR) =================
st.sidebar.title("💼 知識庫配置與狀態")

# 切換系統架構展示
if "show_architecture" not in st.session_state:
    st.session_state.show_architecture = False

if st.sidebar.button("🗺️ 查看系統架構與 RAG 流程圖", use_container_width=True):
    st.session_state.show_architecture = not st.session_state.show_architecture

st.sidebar.subheader("系統狀態")
st.sidebar.markdown(f"**📚 總貼文數（知識庫）**: `{metadata.get('total_posts', 0)}` 篇")
st.sidebar.markdown(f"**💬 總串文數（原始拆分）**: `{metadata.get('total_threads', 0)}` 條")
st.sidebar.markdown(f"**🕒 知識庫更新時間**: `{metadata.get('last_updated', '未知')}`")

st.sidebar.divider()

if st.sidebar.button("🧹 清除對話紀錄", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()

# 檢索與生成參數微調
st.sidebar.subheader("檢索與生成參數")
temperature = st.sidebar.slider(
    "LLM 溫度 (Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="控制生成回答的隨機性。值越低回答越嚴謹、貼合原文。"
)
top_k_rerank = st.sidebar.slider(
    "精篩重排序數量 (Top-K Rerank)",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="第二階段 Cross-Encoder 從候選貼文中重排序選取最相關的前幾篇送給 Gemini。"
)

st.sidebar.divider()

# API Key 配置與檢查
st.sidebar.subheader("API 配置")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key_input = st.sidebar.text_input("請輸入 Gemini API Key:", type="password")
    if api_key_input:
        api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.sidebar.warning("請於此處或環境變數中設定 GOOGLE_API_KEY 才能正常對話。")

# ================= 主頁面 (MAIN PAGE) =================
st.title("💼 Threads 財經與職涯智慧 RAG 問答助理 v2.0")
st.markdown("""
本問答助理採用**雙階段檢索架構 (Dense Retrieval + Cross-Encoder Reranking)**：
1. **第一階段 (初篩/召回)**：使用 `paraphrase-multilingual-MiniLM-L12-v2` 快速檢索 Top-10 候選貼文。
2. **第二階段 (精篩/重排序)**：使用 `CrossEncoder` 進行一對一交叉評分，精選最相關的 Top-3 貼文送至 Gemini。
3. **LLM 生成**：由 `Gemini` 嚴格遵循檢索到的貼文上下文，以原作者（前投行交易員）風格產出繁體中文分析。
""")

st.markdown("""
<div style="background-color: #1e222b; border-radius: 10px; padding: 20px; border: 1px solid #3e4451; margin-bottom: 25px;">
    <h4 style="color: #61afef; margin-top: 0; margin-bottom: 10px;">🎯 知識庫涵蓋領域</h4>
    <p style="font-size: 0.9rem; color: #abb2bf; margin-bottom: 15px;">基於原作者 700 多篇 Threads 貼文的非監督主題分析，本助理已載入以下領域之專業觀點：</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
        <div style="background: #161a22; padding: 12px; border-radius: 8px; border-left: 4px solid #98c379;">
            <strong style="color: #98c379; font-size: 0.95rem;">📈 金融估值與市場分析</strong><br>
            <span style="font-size: 0.85rem; color: #abb2bf; line-height: 1.4;">資產估值邏輯、WACC、利率與匯率傳導、信用利差，以及地緣政治對美債與台積電 EPS 的影響。</span>
        </div>
        <div style="background: #161a22; padding: 12px; border-radius: 8px; border-left: 4px solid #e5c07b;">
            <strong style="color: #e5c07b; font-size: 0.95rem;">🛢️ 原物料交易與機構套利</strong><br>
            <span style="font-size: 0.85rem; color: #abb2bf; line-height: 1.4;">原油近遠月價差與供需框架、汽油裂解價差套利，以及避險基金做空與 MSTR 溢價等交易邏輯。</span>
        </div>
        <div style="background: #161a22; padding: 12px; border-radius: 8px; border-left: 4px solid #c678dd;">
            <strong style="color: #c678dd; font-size: 0.95rem;">💰 理財規劃與資產配置</strong><br>
            <span style="font-size: 0.85rem; color: #abb2bf; line-height: 1.4;">新鮮人靠工作本金翻身的思維、巴菲特保險浮存金模式，以及長期資產配置與退休規劃。</span>
        </div>
        <div style="background: #161a22; padding: 12px; border-radius: 8px; border-left: 4px solid #56b6c2;">
            <strong style="color: #56b6c2; font-size: 0.95rem;">💼 職涯成長與決策邏輯</strong><br>
            <span style="font-size: 0.85rem; color: #abb2bf; line-height: 1.4;">投行分析師職涯探索、流水線教育反思、職場隨機性，以及 coffee chat 轉化為商業資本的方法。</span>
        </div>
        <div style="background: #161a22; padding: 12px; border-radius: 8px; border-left: 4px solid #4b5263;">
            <strong style="color: #abb2bf; font-size: 0.95rem;">🌏 跨國移居與生活觀察</strong><br>
            <span style="font-size: 0.85rem; color: #abb2bf; line-height: 1.4;">澳洲、新加坡與台灣的生活成本、薪資結構與成長環境比較，探討在台灣的生活優勢。</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= 系統架構面板 (ARCHITECTURE PANEL) =================
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
        f'<div style="color:#8892b0;font-size:0.8rem;text-align:right;max-width:52%;">{desc}</div>'
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
        f'<div style="color:#8892b0;font-size:0.8rem;">{subtitle}</div>'
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
        f'<div style="color:#8892b0;font-size:0.68rem;">{sub}</div>'
        '</div>'
    )

def _arrow(label=""):
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;color:#4b5263;font-size:1rem;padding:0 6px;flex-shrink:0;">'
        f'→<span style="font-size:0.62rem;color:#777;">{label}</span></div>'
    )

def _pipe_row(*items):
    return (
        '<div style="display:flex;align-items:center;gap:2px;margin-bottom:12px;flex-wrap:wrap;">'
        + "".join(items) + '</div>'
    )

if st.session_state.show_architecture:
    st.info("💡 **系統架構與檔案組織**：展示專案檔案結構、資料更新管線流程與 RAG 兩階段檢索運作機制。")

    tab_struct, tab_pipe, tab_rag = st.tabs([
        "📁 專案檔案結構",
        "🔄 資料管線流程",
        "🧠 RAG 雙階段檢索"
    ])

    # ── TAB 1: Directory Structure ────────────────────────────────────────────
    with tab_struct:
        st.markdown("#### 📁 專案完整目錄結構")
        st.caption("本專案包含使用者介面、RAG 核心引擎、資料更新管線與儲存檔案：")

        html = (
            '<div style="background:linear-gradient(135deg,#1e222b,#151821);'
            'border:1px solid #3e4451;border-radius:12px;padding:20px;'
            'box-shadow:0 8px 32px rgba(0,0,0,.35);">'
        )
        html += _section_header("🖥️", "使用者介面 (Web UI)", "#c678dd")
        html += _file_row("📄","app.py","前端介面","#c678dd","#c678dd","Streamlit 網頁主程式，負責對話介面渲染、狀態管理與參數調整")

        html += _section_header("⚙️", "核心邏輯 (Core Engine)", "#61afef")
        html += _file_row("📄","rag_engine.py","核心引擎","#61afef","#61afef","RAG 核心運算模組（TextCleaner、VectorIndexer、CrossEncoderReranker、RateLimiter、GeminiGenerator）")

        html += _section_header("🔄", "資料管線 (Data Pipeline)", "#98c379")
        html += _file_row("🚀","main.py","管線主調度","#98c379","#98c379","一鍵自動化管線，依序執行連結收集、內文提取、對話鏈重組與向量建庫")
        html += _file_row("📄","step1_collect_links.py","連結收集","#98c379","#98c379","Playwright 模擬滾動收集貼文 URL，支援歷史比對與早停機制")
        html += _file_row("📄","step2_extract_posts.py","內文提取","#98c379","#98c379","Playwright 爬取各 URL 主貼文與串文，支援 processed_urls 斷點續爬")
        html += _file_row("📄","process_threads_by_post.py","長文重組與建庫","#98c379","#98c379","按順序拼接對話鏈、去重、計算向量並輸出 embeddings_index.pkl")

        html += _section_header("📊", "資料儲存與索引 (Storage)", "#e5c07b")
        html += _file_row("💾","embeddings_index.pkl","向量索引","#e5c07b","#e5c07b","序列化語意向量資料庫（排除模型本體權重，僅約 1.8~3.9 MB）")
        html += _file_row("📊","combined_threads_posts.csv","重組知識庫","#e5c07b","#e5c07b","對話鏈整併後供 RAG 檢索的核心知識庫")
        html += _file_row("📊","threads_posts.csv","原始串文集","#e5c07b","#e5c07b","爬取到的單篇貼文與串文集合")
        html += _file_row("📊","threads_post_links.csv","網址清單","#e5c07b","#e5c07b","收集到的目標貼文網址清單")
        html += _file_row("⚙️","pipeline_metadata.json","統計中繼資料","#e5c07b","#e5c07b","記錄資料庫更新時間、總貼文數與總串文數")

        html += _section_header("📝", "專案文件與配置 (Docs)", "#56b6c2")
        html += _file_row("📝","README.md","說明文件","#56b6c2","#56b6c2","專案使用指南與架構說明")
        html += _file_row("📝","PRD.md","需求規格書","#56b6c2","#56b6c2","產品需求與技術架構規格文件")
        html += _file_row("📝","PROJECT_REPORT.md","成果報告","#56b6c2","#56b6c2","專案技術細節與成果分析報告")
        html += _file_row("📋","requirements.txt","依賴套件","#56b6c2","#56b6c2","Python 環境相依套件清單")
        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

    # ── TAB 2: Data Pipeline Flow ──────────────────────────────────────────────
    with tab_pipe:
        st.markdown("#### 🔄 資料取得與建庫管線流程")
        st.caption("自動化從 Threads 收集貼文、提取內文並整合成語意知識庫的完整流程：")

        pipe_html = (
            '<div style="background:#1a1d25;border:1px solid #3e4451;border-radius:12px;padding:20px;">'
        )
        pipe_html += (
            '<div style="color:#98c379;font-weight:700;margin-bottom:14px;font-size:0.88rem;">'
            '📦 步驟 1：增量收集貼文網址</div>'
        )
        pipe_html += _pipe_row(
            _pipe_box("#98c379","🕸️","step1_collect_links",".py"),
            _arrow("早停機制"),
            _pipe_box("#e5c07b","📊","threads_post_links",".csv"),
        )
        pipe_html += (
            '<div style="color:#61afef;font-weight:700;margin:16px 0 12px;font-size:0.88rem;'
            'border-top:1px solid #3e4451;padding-top:14px;">'
            '🕷️ 步驟 2：貼文提取與斷點續爬</div>'
        )
        pipe_html += _pipe_row(
            _pipe_box("#61afef","🕷️","step2_extract_posts",".py"),
            _arrow("斷點快取"),
            _pipe_box("#56b6c2","💾","processed_urls",".json"),
            _arrow("即時追加"),
            _pipe_box("#e5c07b","📊","threads_posts",".csv"),
        )
        pipe_html += (
            '<div style="color:#c678dd;font-weight:700;margin:16px 0 12px;font-size:0.88rem;'
            'border-top:1px solid #3e4451;padding-top:14px;">'
            '🗂️ 步驟 3：對話鏈重組與向量建庫</div>'
        )
        pipe_html += _pipe_row(
            _pipe_box("#c678dd","⚙️","process_threads_by_post",".py"),
            _arrow("對話鏈拼接"),
            _pipe_box("#e5c07b","📊","combined_threads_posts",".csv"),
            _arrow("向量化"),
            _pipe_box("#e5c07b","💾","embeddings_index",".pkl"),
            _arrow("統計中繼"),
            _pipe_box("#e5c07b","📋","pipeline_metadata",".json"),
        )
        pipe_html += "</div>"
        st.markdown(pipe_html, unsafe_allow_html=True)

        st.markdown("""
**資料管線步驟說明：**
1. **連結收集 (`step1_collect_links.py`)**：使用 Playwright 模擬登入並滾動頁面，若連續遇到 4 篇歷史貼文即觸發早停，輸出 `threads_post_links.csv`。
2. **內文提取 (`step2_extract_posts.py`)**：讀取未處理 URL，抓取主貼文與原作者串文，過濾 UI 雜訊，以 `processed_urls.json` 記錄斷點並即時追加寫入 `threads_posts.csv`。
3. **重組與建庫 (`process_threads_by_post.py`)**：按貼文與串順序排序拼接為完整長文 `combined_threads_posts.csv`，計算 384 維語意向量並序列化為 `embeddings_index.pkl`，最後更新 `pipeline_metadata.json`。
        """)

    # ── TAB 3: RAG Dual-Stage Retrieval ───────────────────────────────────────
    with tab_rag:
        st.markdown("#### 🧠 雙階段檢索與回答生成流程（RAG）")
        st.caption("使用者提問後，系統依序經過以下處理步驟生成回答：")

        rag_steps = [
            ("#2980b9", "🗣️",  "Step 1",  "使用者提問",          "輸入財經或職涯問題"),
            ("#16a085", "🧹",  "Step 2",  "TextCleaner",         "文本清洗與標準化"),
            ("#16a085", "📐",  "Step 3",  "VectorIndexer",       "第一階段：餘弦相似度初篩召回 Top-10"),
            ("#e5c07b", "💾",  "Step 4",  "embeddings_index.pkl","載入本地向量資料庫"),
            ("#16a085", "🔬",  "Step 5",  "CrossEncoderReranker","第二階段：Cross-Encoder 交叉評分精選 Top-K"),
            ("#c0392b", "🛡️",  "Step 6",  "RateLimiter",         "RPM ≤ 10 / RPD ≤ 200 頻率限制防護"),
            ("#d35400", "✨",  "Step 7",  "GeminiGenerator",     "LLM 結合上下文生成繁體中文回覆"),
            ("#2980b9", "💬",  "Step 8",  "Streamlit 介面",      "渲染回答與可展開的參考來源"),
        ]

        rag_html = '<div style="background:#1a1d25;border:1px solid #3e4451;border-radius:12px;padding:20px;">'
        for i, (color, icon, step_label, title, subtitle) in enumerate(rag_steps):
            rag_html += _flow_step(step_label, color, icon, title, subtitle)
            if i < len(rag_steps) - 1:
                rag_html += _connector(color)
        rag_html += "</div>"
        st.markdown(rag_html, unsafe_allow_html=True)

        st.markdown("""
**核心檢索與防護機制：**
1. **第一階段（語意召回）**：`paraphrase-multilingual-MiniLM-L12-v2` 快速計算餘弦相似度，召回 **Top-10** 候選貼文。
2. **第二階段（重排序精篩）**：`cross-encoder/ms-marco-MiniLM-L-6-v2` 進行一對一交叉注意力評分，挑選最相關的 **Top-K（預設 3 篇）** 送交 LLM。
3. **安全防護機制**：內建 **RPM ≤ 10、RPD ≤ 200** 呼叫限制；上下文動態限制在 3000 字元以內，避免超額與過度延遲。
        """)

    st.divider()

# 載入向量索引與重排序模型
indexer = load_indexer(INDEX_PATH)
reranker = load_reranker()

if not indexer:
    st.error("找不到向量索引檔 `embeddings_index.pkl`，請確認是否已執行資料管線建庫。")
else:
    generator = None
    if api_key:
        generator = GeminiGenerator(api_key=api_key)

    if "clicked_prompt" not in st.session_state:
        st.session_state.clicked_prompt = None

    # 渲染對話歷程
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("🔍 查看參考來源與相關性評分"):
                    for i, src in enumerate(msg["sources"]):
                        score_type = 'rerank_score' if 'rerank_score' in src else 'score'
                        st.markdown(f"**[{i+1}] 貼文編號: `{src['post_id']}` (重排序分數: `{src[score_type]:.4f}`)**")
                        st.text_area(f"貼文內容 {i+1}", value=src['document'], height=120, disabled=True, label_visibility="collapsed")

    # 若尚無對話記錄，顯示推薦預設問題
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 推薦範例問題")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 什麼是資產估值的核心邏輯？如何傳導到股債？", use_container_width=True):
                st.session_state.clicked_prompt = "什麼是資產估值的核心邏輯？如何傳導到股債？"
                st.rerun()
            if st.button("💰 新鮮人月薪 30K 該如何理財翻身與配置？", use_container_width=True):
                st.session_state.clicked_prompt = "新鮮人月薪 30K 該如何理財翻身與配置？"
                st.rerun()
        with col2:
            if st.button("💼 投行分析師如何看待職涯轉型與人生隨機性？", use_container_width=True):
                st.session_state.clicked_prompt = "投行分析師如何看待職涯轉型與人生隨機性？"
                st.rerun()
            if st.button("🛢️ 俄烏戰爭中如何操作汽油「裂解價差套利」？", use_container_width=True):
                st.session_state.clicked_prompt = "俄烏戰爭中如何操作汽油「裂解價差套利」？"
                st.rerun()

        if st.button("🌏 比較澳洲、新加坡與台灣的生活與孩子成長環境？", use_container_width=True):
            st.session_state.clicked_prompt = "比較澳洲、新加坡與台灣的生活與孩子成長環境？"
            st.rerun()

    # 接收使用者輸入
    user_query = st.chat_input("請輸入您的財經或職涯問題...")

    if st.session_state.clicked_prompt:
        user_query = st.session_state.clicked_prompt
        st.session_state.clicked_prompt = None

    if user_query:
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
                    # 階段一：語意初篩 (召回 Top-10)
                    top_10_candidates = indexer.search(user_query, top_k=10)

                    # 階段二：Cross-Encoder 重排序 (精篩 Top-K)
                    final_retrieved = reranker.rerank(user_query, top_10_candidates, top_k=top_k_rerank)

                    # 階段三：動態字元長度限制 (限制 3000 字元以內)
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

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": final_retrieved_docs
                        })

                        with st.expander("🔍 查看參考來源與相關性評分"):
                            for i, src in enumerate(final_retrieved_docs):
                                st.markdown(f"**[{i+1}] 貼文編號: `{src['post_id']}` (重排序分數: `{src['rerank_score']:.4f}`)**")
                                st.text_area(f"貼文內容 {i+1}", value=src['document'], height=120, disabled=True, label_visibility="collapsed")

                    except Exception as e:
                        st.error(f"對話生成出錯: {str(e)}")
