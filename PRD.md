# 產品需求規格書 (PRD)：Threads 財經與職涯問答助理 v2.0

---

## 1. 產品概述 (Executive Summary)

### 1.1 背景
第一代 (v1.0) 系統已透過實驗驗證「合併貼文 (Merged Flow)」能保留原作者（前投資銀行交易員）在 Threads 上的財經與職涯分析脈絡，並透過檢索增強生成 (RAG) 結合 Gemini 生成回覆。

### 1.2 目標
v2.0 旨在將實驗腳本升級為具備自動化更新機制與兩階段檢索的互動式 Web 應用程式：
1. **資料維護自動化**：建立支援增量檢查、早停機制與斷點續爬的資料更新管線。
2. **檢索精準度提升**：導入「向量粗篩 + Cross-Encoder 重排序」兩階段檢索，降低長文檢索誤差與 Token 消耗。
3. **介面化互動**：使用 Streamlit 建置聊天介面，提供透明的參考來源展示與 API 呼叫頻率保護。

---

## 2. 核心目標與指標 (Goals & KPIs)

* **易用性目標**：提供無須撰寫程式碼即可使用的 Web 對話介面，支援 Markdown 格式渲染與參考來源折疊面板。
* **準確性目標**：透過 Cross-Encoder 重排序提升送入 LLM 的上下文關聯度，降低模型幻覺。
* **自動化目標**：提供一鍵調度腳本，自動完成爬取、去重、串文重組與向量索引更新。
* **效益指標 (KPIs)**：
    * **回應延遲**：從使用者發送問題到產生回覆的時間控制在 5–8 秒以內。
    * **上下文優化**：透過重排序將候選貼文從 Top-10 精簡為 Top-3 送交 LLM，降低 Token 消耗約 70%。
    * **更新效率**：增量爬蟲搭配早停機制，於無大量新貼文時大幅縮減執行時間。

---

## 3. 功能需求規格 (Functional Requirements)

### 3.1 互動式 Web 介面 (Streamlit)
* **對話聊天區**：
    * 提供使用者輸入框與對話氣泡，支援繁體中文 Markdown 語法解析。
* **參考來源折疊面板 (Source Accordion)**：
    * 每則回覆下方提供「查看參考來源與相關性評分」展開區塊，顯示引用的貼文編號、文字內容與重排序分數。
* **側邊欄控制項**：
    * **狀態顯示**：即時讀取 `pipeline_metadata.json` 顯示知識庫總篇數、總串文數與最後更新時間。
    * **參數調整**：提供滑桿調整生成溫度 (Temperature, 預設 0.2) 與重排序選取數量 (Top-K Rerank, 預設 3)。
    * **對話歷程管理**：提供清除對話紀錄按鈕。
    * **API 金鑰配置**：支援由環境變數讀取或在前端輸入框填入 Gemini API Key。

### 3.2 兩階段檢索架構 (Dense Retrieval + Re-ranking)
* **第一階段（語意召回 - 粗篩）**：
    * 模型：`paraphrase-multilingual-MiniLM-L12-v2` (Bi-Encoder)。
    * 計算使用者提問向量與知識庫向量之餘弦相似度，召回 Top-10 候選貼文。
* **第二階段（深度重排序 - 精篩）**：
    * 模型：`cross-encoder/ms-marco-MiniLM-L-6-v2` (Cross-Encoder)。
    * 將使用者提問與 10 篇候選貼文進行一對一交叉比對評分，依關聯度分數選取 Top-3 送交 LLM。

### 3.3 安全防護與生成控制
* **API 頻率限制 (Rate Limiter)**：
    * 監控請求時間戳記，設定每分鐘請求數 (RPM ≤ 10) 與每日請求數 (RPD ≤ 200)。
    * 尖峰時段自動暫停等待，超額時回傳提示訊息。
* **上下文長度截斷**：
    * 傳送給 LLM 前動態限制上下文總字元數在 3000 字以內，避免超額與延遲增加。
* **提示詞約束 (Prompt Constraint)**：
    * 限定模型以作者視角與專業風格回答。
    * 要求必須使用繁體中文。
    * 嚴格規定若檢索貼文無相關內容，須如實告知未找到，禁止使用外部知識編造。

### 3.4 自動化資料更新管線 (Data Pipeline)
* **主調度器 (`main.py`)**：
    * 提供命令列參數 `--user`、`--cookie` 與 `--threshold`，非同步串接三步驟流程，並產出摘要報告 `pipeline_summary.txt`。
* **步驟 1：連結收集 (`step1_collect_links.py`)**：
    * 使用 Playwright 載入 `cookies.json` 模擬登入並滾動頁面。
    * 比對現有 `threads_post_links.csv`，連續命中 4 篇舊貼文即觸發早停。
* **步驟 2：內文提取 (`step2_extract_posts.py`)**：
    * 讀取未處理 URL，抓取主貼文與作者後續串文，過濾非正文 UI 字詞。
    * 透過 `processed_urls.json` 支援斷點續爬，並逐篇追加寫入 `threads_posts.csv`。
* **步驟 3：長文重組與建庫 (`process_threads_by_post.py`)**：
    * 依貼文編號與串文序號排序拼接為連貫長文，輸出 `combined_threads_posts.csv`。
    * 執行文本清洗並使用 `VectorIndexer` 計算 384 維向量，輸出 `embeddings_index.pkl` 與 `pipeline_metadata.json`。

---

## 4. 系統流程架構

```
[ 執行 main.py 管線主程式 ]
         │
         ▼
 1. step1_collect_links.py (增量收集 + 早停) ───▶ threads_post_links.csv
         │
         ▼
 2. step2_extract_posts.py (斷點續爬 + 逐篇追加) ─▶ threads_posts.csv & processed_urls.json
         │
         ▼
 3. process_threads_by_post.py (重組拼接 + 向量化) ─▶ combined_threads_posts.csv
         │                                            embeddings_index.pkl
         │                                            pipeline_metadata.json
         │
         │ (Streamlit app.py 載入)
         ▼
[ 使用者於 Streamlit 輸入提問 ] ──────────────────▶ 4. VectorIndexer 初篩 (召回 Top-10)
                                                      │
                                                      ▼
[ Gemini 結合上下文生成回覆 ] ◀──────────────────── 5. CrossEncoder 重排序 (精篩 Top-3)
         │                                            (包含 RateLimiter 與 3000 字元截斷)
         ▼
[ 介面呈現回答與可展開參考來源 ]
```

---

## 5. 技術架構與套件

| 模組領域 | 核心技術 / 套件 | 說明 |
|---|---|---|
| **網頁介面** | Python + Streamlit (`streamlit`) | 互動式對話介面與狀態管理 |
| **瀏覽器自動化** | `playwright`, `beautifulsoup4` | 模擬登入、頁面滾動與 HTML 結構解析 |
| **語意向量檢索** | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | 384 維多語言密集向量運算 |
| **精篩重排序** | `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | 交叉比對關聯度評分 |
| **大型語言模型** | `google-generativeai` (Gemini Flash / Flash Lite) | 上下文理解與文本生成 |
| **資料處理與儲存** | `pandas`, `numpy`, `pickle`, `json`, `re` | 文本清洗、對話鏈拼接、快取與序列化 |