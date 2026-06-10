1. 產品概述 (Executive Summary)
  背景：
  第一代 (v1.0) 系統已透過 Jupyter Notebook 驗證「合併貼文 (Merged Flow)」能有效保留原作者的財經與職涯思維脈絡，且 Gemini 生成品質良好。
  目標：
  v2.0 旨在將本地端腳本升級為自動化、高精準度的互動式 Web 應用程式。透過建立資料自動更新管線解決資料過時問題，並導入重排序 (Re-ranking) 技術解決多篇長文檢索的精準度與 Token
  消耗問題，最終透過 Web 介面提供流暢的使用者體驗。

  ---

  2. 核心目標與成功指標 (Goals & KPIs)
   * 目標 1 (產品化)： 提供無須寫程式即可互動的 Chatbot 網頁介面。
   * 目標 2 (高精準)： 提升 RAG 檢索的「信噪比」，確保交給 LLM 的上下文是最核心的段落，降低幻覺與 Token 浪費。
   * 目標 3 (自動化)： 減少人工手動匯出/清洗資料的負擔，實現一鍵（或定期）更新知識庫。
   * 成功指標 (KPIs)：
       * 使用者提問到產生回答的延遲時間 (Latency) < 5-8 秒。
       * 檢索命中率提升：透過 Re-ranking 確保 Top-3 文檔的關聯性大於單純向量檢索的 Top-3。

  ---

  3. 核心功能需求 (Key Features)

  功能一：互動式 Web 介面 (基於 Streamlit)
   * 使用者介面 (UI)：
       * 對話區 (Chat Interface)： 仿照 ChatGPT 的對話框設計，支援 Markdown 語法渲染（粗體、條列式），呈現 Gemini 生成的專業財經分析。
       * 參考來源展開 (Source Toggle)： 在回答下方提供「查看參考貼文」的折疊選單 (Accordion)，讓使用者能點開查閱原始 Threads 貼文內容與檢索相似度分數，建立信任感。
       * 側邊欄控制 (Sidebar控制項)：
           * 系統狀態指示器（顯示目前知識庫最後更新時間、總貼文數）。
           * 進階設定（可選）：允許調整生成溫度 (Temperature) 或 Re-ranking 輸出的 Top-K 數量。

  功能二：兩階段高精準檢索架構 (Dense Retrieval + Re-ranking)
   * 痛點解決： 合併貼文 (Merged Flow) 字數較長，若單純依賴 Sentence-Transformer 算 Cosine Similarity，容易受局部字詞影響，且一次丟太多篇給 Gemini 會導致 Context 過長。
   * 架構設計：
       * 第一階段 (粗篩 - 召回)： 使用目前的 paraphrase-multilingual-MiniLM-L12-v2，針對使用者問題快速檢索出 Top-10 篇最相關的候選貼文。
       * 第二階段 (精篩 - 重排序)： 導入 Cross-Encoder 模型（如 cross-encoder/ms-marco-MiniLM-L-6-v2 或 BGE-Reranker）。讓模型將「使用者問題」與「這 10
         篇候選文檔」一對一交叉比對，產出更精確的相關性分數。
       * 輸出： 最終只取 Re-ranking 分數最高的 Top-2 或 Top-3 交給 Gemini 生成回答。

  功能三：自動化資料更新管線 (Automated Data Pipeline)
   * 痛點解決： 目前資料需手動爬蟲、手動清洗。
   * 架構設計： 撰寫一支獨立的 update_pipeline.py 腳本，將流程自動化串接：
       1. 觸發： 支援手動執行指令（或未來設定 Cronjob 定期執行）。
       2. 爬取： 自動呼叫 custom_threads_scraper.py 抓取最新貼文。
       3. 整併與清洗： 執行 step2_extract_posts.py 與 merge_posts.py，並套用 Notebook 中的 clean_text 函數去除雜訊。
       4. 向量更新： 重新計算新貼文的 Embedding，並將合併後的資料集與向量索引儲存到本地（如更新 CSV 並儲存 .pkl 或 .npy 索引檔）。
       5. 熱更新： Streamlit 偵測到資料檔更新後，自動重新載入最新知識庫。

  ---

  4. 系統架構流程圖 (System Architecture)

    1 [ 定期 / 手動觸發 Pipeline ]
    2        │
    3        ▼
    4 1. 爬蟲腳本抓取新 Threads 貼文
    5        │
    6        ▼
    7 2. 文本清洗與貼文合併 (Merged Flow 邏輯)
    8        │
    9        ▼
   10 3. Sentence-Transformer 計算向量 ───▶ 儲存至本地索引 (Index.pkl & CSV)
   11                                            │
   12                                            │ (Streamlit App 載入)
   13                                            ▼
   14 [ 使用者在 Web 介面輸入提問 ] ─────────▶ 4. 語意向量初篩 (召回 Top-10)
   15                                            │
   16                                            ▼
   17 [ Gemini 結合上下文生成專業回答 ] ◀──── 5. Cross-Encoder 重排序 (精篩 Top-3)
   18        │
   19        ▼
   20 [ Web 介面渲染回答與參考來源 ]

  ---

  5. 開發階段與技術棧 (Development Phases & Tech Stack)

  技術棧：
   * 前端/框架： Python + Streamlit (streamlit)
   * 檢索與重排序： sentence-transformers (Bi-encoder + Cross-encoder)
   * LLM 引擎： google-genai (Gemini 3.1 Flash Lite 或 Flash)
   * 資料處理： pandas, numpy

  開發時程 (Phases)：
   * Phase 1：實作重排序 (Re-ranking) 引擎
       * 在原有的 Jupyter Notebook 中加入 Cross-Encoder 邏輯，驗證 Top-10 縮減到 Top-3 的精準度提升效果。
   * Phase 2：封裝 Web 應用程式
       * 建立 app.py，將 VectorIndexer、RAGEngine 封裝為 Class。
       * 使用 Streamlit 建立 Chat UI，並將 Phase 1 的 RAG 邏輯接入。
   * Phase 3：自動化管線串接
       * 編寫 update_pipeline.py，將爬蟲 ➡️ 清洗 ➡️ 存檔的腳本整合。確保 Streamlit 能透過 st.cache_resource 讀取最新的索引檔。