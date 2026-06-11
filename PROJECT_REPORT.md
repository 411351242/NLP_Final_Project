# 📊 專案成果報告：Threads 財經與職涯智慧問答助理 v2.0

> **文件對象**：非技術背景的主管、評審委員、合作夥伴
> **文件目的**：說明本專案使用的工具、每個步驟的具體做法，以及最終產出的價值
> **線上展示**：https://nlpfinalproject-nnrg9joqc32rdajxyyeks3.streamlit.app/

---

## 📌 一、專案背景與動機

### 問題的起點

本專案的知識來源，是一位曾在國際投資銀行擔任交易員的作者在 Threads 平台上長期分享的財經與職涯觀點，內容包含：

- 💰 投資策略與風險管理思維（如日圓套利、資產負債表分析）
- 📈 總體經濟觀察（如央行政策、債市利率）
- 🧭 金融業職涯規劃（如薪資架構、晉升思維）
- 🧠 個人理財與高效工作哲學

然而，這些貼文**散落在社群平台的數百則貼文中**，要找到特定觀點需要大量時間翻閱。

### 核心挑戰

| 問題 | 說明 |
|---|---|
| 🔍 資訊散落 | 數百篇貼文無法快速搜尋特定主題 |
| 🤖 AI 容易亂答 | ChatGPT 等通用 AI 會「憑空捏造」不屬於作者的觀點 |
| 🔗 串文切碎 | 作者的長文常分多則發佈，單篇閱讀容易斷章取義 |
| 📅 資料會過時 | 作者持續發新文，手動更新非常耗時 |

---

## 📦 三、模組一：資料管線

**負責程式**：`step1_collect_links.py`、`step2_extract_posts.py`、`merge_posts.py`、`process_threads_by_post.py`

這個模組的任務是：把作者在 Threads 上的貼文，從原始社群資料，整理成 AI 可以使用的乾淨對話鏈知識庫。整個流程共分四個步驟。

---

### 步驟 1：連結收集（`step1_collect_links.py`）

**使用套件**：`playwright`（Playwright 瀏覽器自動化）、`pandas`

**做了什麼**：
- 載入 `cookies.json` 憑證以模擬登入狀態。
- 自動化瀏覽器開啟作者首頁 `https://www.threads.net/@make_investment_easy`。
- 漸進式向下滾動，即時收集所有匹配 `/post/` 格式的貼文連結。
- 排除重複與無效連結，將收集到的 URL 寫入 **`threads_post_links.csv`**。

---

### 步驟 2：貼文爬取（`step2_extract_posts.py`）

**使用套件**：`playwright`（Playwright 瀏覽器自動化）、`beautifulsoup4`、`pandas`

**做了什麼**：
1. 讀取步驟一產生的 `threads_post_links.csv`。
2. 逐一巡覽貼文網頁，抓取主貼文與原作者的所有後續串文（排除其他使用者的回覆）。
3. 使用 `clean_post_text` 進行初步 UI 噪訊清理（過濾作者名稱、按讚、分享、時間等元件文字）。
4. 為了防止因意外中斷遺失進度，將爬取的貼文**分批（預設 50 篇一組）匯出為 `threads_posts-XX.csv`**。

---

### 步驟 3：貼文合併（`merge_posts.py`）

**使用套件**：`pandas`

**做了什麼**：
- 偵測資料夾內所有 `threads_posts-XX.csv` 分批檔案。
- 將讀取到的 DataFrames 進行合併，並且進行內容去重（以貼文內容為關鍵字），輸出為完整的原始貼文檔案 **`threads_posts.csv`**。

---

### 步驟 4：對話鏈重組與清洗（`process_threads_by_post.py`）

**使用套件**：`pandas`、`re`（正規表達式）

**做了什麼**：
- 讀取合併後的 `threads_posts.csv`。
- 使用正規表達式解析出「貼文編號」與「串文順序」，並依數值進行排序，確保語句邏輯連貫。
- 清洗並移除貼文尾端的平台結尾噪訊（如「(續」、「續」等標記）。
- 按主貼文 ID 進行 `groupby`，將同個對話主題的所有後續串文**按順序以換行符號連接**，重組為連貫的完整上下文，輸出為 **`combined_threads_posts.csv`**，作為知識庫的最終檢索來源。

---

### 📊 語意向量化與建庫（`VectorIndexer`）

為了供 RAG 核心高速檢索，系統載入 `combined_threads_posts.csv` 並進行向量化：

#### A. 文字清洗
在向量化前，再度清除 Unicode 亂碼、網址、Hashtag、@Tag 等文字，並使用 `TextCleaner.clean_text` 進行中英文間距壓縮。

#### B. 語意向量化（`VectorIndexer.fit()`）
使用多語言預訓練模型 `paraphrase-multilingual-MiniLM-L12-v2`，將每篇重組後的對話鏈轉換為向量索引，代表其語意位置。此時 738 組向量會被當作檢索索引載入。

#### C. 索引序列化存檔
將向量索引物件序列化儲存成 **`embeddings_index.pkl`**（僅 1.81 MB，已透過自訂排除 PyTorch 模型權重），並寫入統計資料至 **`pipeline_metadata.json`**，供網頁介面即時顯示。

---

## 🧠 四、模組二：RAG 雙階段高精準檢索引擎

**負責程式**：`rag_engine.py`

### RAG 是什麼？

**RAG（Retrieval-Augmented Generation，檢索增強生成）** 是一種讓 AI 「先查資料、再回答」的技術架構，有效解決 AI 憑空捏造的問題。

```
一般 AI（如 ChatGPT）：
  你問 → AI 從訓練記憶中組合答案 → 可能答非所問或捏造

本系統 RAG：
  你問 → 找最相關的作者貼文 → AI 根據貼文回答 → 答案有所本、可追溯
```

### 為什麼使用「兩階段」？

單一階段的向量搜尋對長文容易失準（局部關鍵字可能蓋過整體語意）。兩階段設計讓系統「先廣泛召回、再精準篩選」：

| | 第一階段（粗篩） | 第二階段（精篩） |
|---|---|---|
| 名稱 | Dense Retrieval（密集向量召回）| Re-ranking（重排序）|
| 工具 | Bi-Encoder | Cross-Encoder |
| 速度 | 快（毫秒級） | 較慢（需逐一比對）|
| 輸出 | Top-10 候選貼文 | Top-3 最終貼文 |

---

### 第一階段：語意召回（`VectorIndexer.search()`）

**使用套件**：`sentence_transformers`（`SentenceTransformer.encode()`）、`sklearn.metrics.pairwise`（`cosine_similarity()`）、`numpy`（`argsort()`）

**做了什麼**：

1. 用 `pickle.load()` 從磁碟載入預先建好的 `embeddings_index.pkl`
2. 使用 `SentenceTransformer.encode()` 將**使用者的提問**也轉換成一組向量
3. 用 `sklearn` 的 `cosine_similarity()` 計算**使用者問題向量**與**738 篇貼文向量**的餘弦相似度（數值越接近 1，代表語意越相近）
4. 用 `numpy.argsort()` 排序，取出相似度最高的 **Top-10 候選貼文**

> 💡 **比喻**：就像在圖書館用關鍵字找書，先把可能相關的 10 本書都拿出來。

---

### 第二階段：精篩重排序（`CrossEncoderReranker.rerank()`）

**使用套件**：`sentence_transformers`（`CrossEncoder.predict()`）

**做了什麼**：

1. 用 `CrossEncoder` 類別載入 **`cross-encoder/ms-marco-MiniLM-L-6-v2`** 模型
2. 將**使用者問題**與 **Top-10 每篇候選貼文**一對一組成文字對（共 10 對）
3. 呼叫 `CrossEncoder.predict()` 對每一對進行**深度交叉比對評分**——模型同時讀問題和文章，給出一個 0~10+ 的相關性分數
4. 按分數由高到低排序，取出 **Top-K（預設 3）篇**最相關的貼文

> 💡 **比喻**：就像從 10 本候選書中，請一位專家逐本對照你的問題深度審閱，最終挑出最切題的 3 本。

---

### 文字安全閥（動態截斷）

**使用套件**：Python 內建 `len()` 函數

在把貼文傳給 Gemini 之前，系統會累計所有文字的**字元數**，若超過 **3000 字**（約 2000 tokens）就停止新增，確保：
- 回應延遲控制在 **5–8 秒**以內
- 不超出 Gemini API 的 Context 限制

---

### LLM 生成回答（`GeminiGenerator.generate()`）

**使用套件**：`google.generativeai`（`GenerativeModel.generate_content()`）

**做了什麼**：

1. 用 `genai.configure(api_key=...)` 設定 Gemini API 金鑰
2. 初始化 `GenerativeModel('gemini-3.1-flash-lite-preview')` 模型實例
3. 組裝 **Prompt（提示詞）**，內容包含：
   - 角色設定（模擬投銀交易員口吻、只能用繁體中文、禁止使用外部知識）
   - Top-3 篇貼文作為參考上下文
   - 使用者的原始問題
4. 呼叫 `GenerativeModel.generate_content()` 傳送至 Gemini API，取得生成的回答文字

**Prompt 的嚴格限制（防止 AI 亂說話）**：
- 「若參考貼文中完全沒有相關資訊，請坦白回答『未找到相關論述』，絕對不要憑空編造」
- 「必須使用繁體中文」
- 「回答應分點條列，引用作者的邏輯框架」

---

### API 慢速保護（`RateLimiter.check_and_log()`）

**使用套件**：`json`（`json.load()`、`json.dump()`）、`time`（`time.time()`、`time.sleep()`）

**做了什麼**：

在每次呼叫 Gemini 之前，系統都會先執行以下檢查：

1. 用 `json.load()` 讀取 `rate_limit_log.json`，取得過去所有 API 呼叫的時間戳記清單
2. 計算**過去 1 分鐘內**的請求數（RPM 檢查）
3. 計算**過去 24 小時內**的請求數（RPD 檢查）
4. 若 RPM ≥ 10，用 `time.sleep()` 自動暫停等待（讓頻率降回限制以下）
5. 若 RPD ≥ 200，直接回傳警示訊息，當日不再呼叫 API
6. 通過檢查後，把本次時間戳記用 `json.dump()` 寫入紀錄檔

> 🛡️ **目的**：確保公開展示期間，Gemini API 金鑰不會因過度使用而爆炸或停權。

---

## 🖥️ 五、模組三：Streamlit 網頁介面

**負責程式**：`app.py`

**使用套件**：`streamlit`（多個元件方法）、`pickle`（`pickle.load()`）

### 頁面初始化

- `st.set_page_config()` 設定頁籤標題、圖示和版面（寬螢幕模式）
- `st.markdown()` 注入自訂 CSS 樣式，打造深色質感介面
- `load_dotenv()` 從 `.env` 檔案讀取環境變數（Gemini API Key、管理員密碼）

### 快取載入（避免重複計算）

- `@st.cache_resource` 裝飾器：確保 `embeddings_index.pkl` 和 Cross-Encoder 模型**只在啟動時載入一次**，之後重新整理頁面不會重新載入，節省大量時間
- `pickle.load()` 從磁碟還原向量索引物件

### 側邊欄（Sidebar）

- `st.sidebar.subheader()` 分區標題
- `st.sidebar.markdown()` 顯示知識庫統計（貼文數、串文數、更新時間），數據來自 `json.load()` 讀取的 `pipeline_metadata.json`
- `st.sidebar.text_input(type="password")` 管理員密碼輸入框
- `st.sidebar.button()` 觸發「一鍵更新知識庫」按鈕
- `st.sidebar.slider()` 提供 Temperature（AI 創造性）和 Top-K Rerank（精篩數量）的參數調整
- 管理員驗證通過後，`subprocess.run([sys.executable, "update_pipeline.py"])` 在背景執行整個資料更新管線
- `st.cache_resource.clear()` + `st.rerun()` 更新完成後清除快取並刷新頁面

### 架構展示面板

- `st.button()` 觸發系統架構展開/收起
- `st.tabs()` 建立三個頁籤（檔案結構、資料管線、RAG 檢索）
- `st.markdown(unsafe_allow_html=True)` 渲染自訂 HTML/CSS 卡片與流程圖

### 聊天介面

- `st.session_state` 字典儲存對話歷程，確保跨次問答的上下文連貫
- `st.chat_input()` 提供底部輸入框
- `st.chat_message("user")` / `st.chat_message("assistant")` 顯示對話氣泡
- `st.spinner()` 在等待時顯示「處理中」提示
- `st.expander()` 折疊展開「查看參考來源與相關性評分」詳情

---

## 📊 六、量化成果

| 指標 | 數值 | 說明 |
|---|---|---|
| 📚 知識庫貼文數 | **738 篇** | 最終整併後的高品質繁中與雙語貼文 |
| 📝 原始串文數 | **1693 串** | 經過去重處理後的原始單篇貼文總數 |
| 💾 索引檔大小 | **1.81 MB** | 優化後縮減 **250 倍**（原始為 459 MB）|
| ⏱️ 回答延遲 | **5–8 秒** | 符合 PRD 產品目標 |
| 🎯 精篩貼文數 | Top-10 → Top-3 | 相關性更高，Token 消耗減少 70% |
| 🛡️ API 每分鐘上限 | **10 次（RPM）** | 慢速模式保護 |
| 🛡️ API 每日上限 | **200 次（RPD）** | 展示期間費用可控 |

---

## 🗂️ 七、檔案結構與角色分工

| 檔案 | 類型 | 角色 | 核心套件 |
|---|---|---|---|
| `app.py` | Python | 網頁介面主程式 | `streamlit` |
| `rag_engine.py` | Python | RAG 核心引擎（清洗 / 向量 / 重排序 / 限速）| `sentence_transformers`、`sklearn`、`google.generativeai` |
| `step1_collect_links.py` | Python | 貼文連結收集腳本 | `playwright`、`pandas` |
| `step2_extract_posts.py` | Python | 貼文內容爬取腳本 | `playwright`、`beautifulsoup4`、`pandas` |
| `merge_posts.py` | Python | 貼文初步整併與去重 | `pandas` |
| `process_threads_by_post.py` | Python | 串文對話鏈重組與清洗 | `pandas`、`re` |
| `combined_threads_posts.csv` | 資料 | 最終整併對話鏈知識庫（738 篇）| — |
| `threads_posts.csv` | 資料 | 原始單篇貼文集（1693 條）| — |
| `embeddings_index.pkl` | 資料 | 向量索引序列化檔 | `pickle` |
| `pipeline_metadata.json` | 資料 | 知識庫狀態與統計中繼資料 | `json` |
| `requirements.txt` | 配置 | Streamlit Cloud 部署依賴套件 | — |
| `PRD.md` | 文件 | 產品需求規格文件 | — |
| `preprocessing_recommendations.md` | 文件 | 資料前處理優化建議書 | — |

---

## 🔄 八、v1.0 到 v2.0 的演進

| 面向 | v1.0（Jupyter Notebook 本地版）| v2.0（Web 應用正式版）|
|---|---|---|
| 操作方式 | 需執行 Python 程式碼 | 瀏覽器直接聊天 |
| 部署方式 | 本機執行 | Streamlit Cloud 公開部署 |
| 知識庫更新 | 手動執行多個腳本 | 本機順序執行取得資料腳本 |
| 檢索架構 | 單階段向量搜尋 | 雙階段（召回 + Cross-Encoder 重排序）|
| Token 防護 | 無 | 動態截斷（≤ 3000 字）|
| API 防護 | 無 | RPM ≤ 10、RPD ≤ 200 限速 |
| 隱私保護 | Cookie 風險未處理 | 密碼保護 + Cookie 本地隔離 |
| 索引檔大小 | 459 MB（含模型權重）| 1.81 MB（排除模型權重）|

---

## ❓ 九、常見問題

**Q：系統的回答都是真的嗎？有沒有可能胡說八道？**

A：系統設有**嚴格的 Prompt 防護**，指令要求 AI「若貼文中沒有相關資訊，必須如實告知未找到，絕對不可捏造」。每則回答下方也有「查看參考來源」功能，可直接對照原文驗證。

**Q：任何人都可以修改知識庫嗎？**

A：不行。知識庫更新功能受**管理員密碼**（`ADMIN_PASSWORD`）保護，且真實的爬蟲憑證（Cookie）只存在管理員本機，不存在公開伺服器上。

**Q：這套系統能否用於其他主題？**

A：可以。只要替換知識庫（將 CSV 換成其他文件，例如公司內部規章、法規手冊），並重新執行向量化建庫，整個 RAG 架構即可複用於任何垂直領域。

---

## 📬 結語

本專案從社群平台的財經知識散落問題出發，完整實現了：

1. **自動化資料管線**：爬取 → 去重 → 串文重組 → 清洗 → 向量化，全流程一鍵完成
2. **雙階段高精準 RAG**：`sentence_transformers` 向量召回 + `CrossEncoder` 精篩，大幅提升回答品質
3. **生產等級防護**：API 慢速限速、Context 動態截斷、密碼保護、Cookie 隔離
4. **無程式碼的使用者體驗**：`Streamlit` 聊天介面，任何人皆可即時使用

---

*最後更新：2026 年 6 月*
*專案開發：Kevin（自然語言處理 Final Project）*
