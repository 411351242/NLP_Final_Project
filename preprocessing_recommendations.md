# Threads 繁體中文社群 RAG 資料前處理優化建議與應用指引

本文件針對 Threads 財經與職涯智慧語料庫的特性（包含口語化、多表情符號、多段串文、高雜訊等），評估適合整合至 RAG 問答助理（[rag_assistant.ipynb](file:///c:/Users/Kevin/Desktop/NEWFILE/NLP/FinalProject/rag_assistant.ipynb)）中的資料前處理方法與具體應用指引。

---

## 📌 適合引入之核心前處理方法

### 1. 文本切分與滑動視窗 (Document Chunking & Sliding Window)
* **目的與優勢**：
  在 RAG 中，**「流程 A：合併貼文集」** 雖能保證思維脈絡的完整性，但少數合併後的長貼文（例如超過 2000 字）如果直接作為上下文（Context）送給 LLM，會稀釋語意表徵的精準度，且容易使 Top-K 檢索結果突破 LLM 的 Context Limit。
  導入基於字元數的滑動視窗（Sliding Window with Overlap）能將文本切分為適中區塊（如 600 字），並透過重疊字元（如 150 字）確保斷句處語意不遺失。
* **具體應用方式**：
  在數據載入與基本清洗後，針對長貼文呼叫切分器，將切分後的段落與對應的 `貼文編號` 關聯並寫入向量索引。
* **實作程式碼範例**：
  ```python
  def split_text_with_overlap(text, chunk_size=600, overlap=150):
      """字元級滑動視窗切分（適用於繁體中文）"""
      if len(text) <= chunk_size:
          return [text]
      chunks = []
      start = 0
      while start < len(text):
          end = start + chunk_size
          chunks.append(text[start:end])
          start += (chunk_size - overlap)
      return chunks
  ```

### 2. 語言偵測與語系過濾 (Language Detection & Filtering)
* **目的與優勢**：
  社群平台（如 Threads）的貼文包含大量外文廣告、多語系垃圾內容或純網址。這類噪訊會嚴重干擾語意向量模型（如 `paraphrase-multilingual-MiniLM-L12-v2`）的計算。利用語言偵測，能確保進入知識庫的語料皆為高質量的繁體中文。
* **具體應用方式**：
  在載入 CSV 數據後，利用 `langdetect` 進行語系檢測，僅保留語系為 `zh-tw`、`zh-cn`（中文）或特定的英文內容。
* **實作程式碼範例**：
  ```python
  from langdetect import detect, DetectorFactory
  DetectorFactory.seed = 0 # 保證偵測結果一致

  def filter_chinese_posts(df, text_col='文字內容'):
      """過濾非中文或無法識別之噪訊貼文"""
      def is_chinese(text):
          try:
              lang = detect(text)
              return lang in ['zh-tw', 'zh-cn', 'cn']
          except:
              return False
      return df[df[text_col].apply(is_chinese)].copy()
  ```

### 3. 混合檢索中的中文斷詞與停用詞移除 (Chinese Segmentation & Stopwords)
* **⚠️ 關鍵原則：向量與 LLM 生成時「不可移除」，關鍵字檢索「必須移除」**：
  * **向量模型與 LLM**： modern embedding 模型與 Gemini 生成需要豐富的自然語言上下文、標點符號與語氣詞（如 `XD`、`@@`），用以精準捕捉作者專屬的「固收交易員口吻」。此時**絕對不可**過濾停用詞與標點。
  * **稀疏檢索（如 BM25）**：若未來引入雙階段或混合檢索，關鍵字檢索（Sparse Retrieval）需要過濾無意義高頻詞（如 "的"、"了"、"在"）。
* **具體應用方式**：
  使用 `jieba` 分詞器，並加載繁體中文停用詞表，在建立 BM25 索引或計算關鍵字詞頻前進行清洗。
* **實作程式碼範例**：
  ```python
  import jieba

  CHINESE_STOPWORDS = set(["的", "了", "在", "是", "我", "你", "他", "與", "及", "等", "續", "Read more"])

  def tokenize_for_bm25(text):
      """專為關鍵字 BM25 檢索設計的分詞與過濾"""
      words = jieba.cut(text)
      return [w for w in words if w.strip() and w not in CHINESE_STOPWORDS]
  ```

### 4. 斷句與句子視窗檢索 (Sentence Splitting & Sentence-Window Retrieval)
* **目的與優勢**：
  有時使用者提問與文檔中的某個「特定句子」最為相似。若以整個段落進行向量匹配，特徵容易被稀釋。以「句子」為單位建庫，但在檢索到最相似的句子後，動態擴展該句的前後幾句（Window）送給 LLM，能大幅提高檢索精準度，同時保證回答內容有足夠上下文。
* **具體應用方式**：
  預處理時使用正則表達式對段落進行中文標點斷句，建立句子級向量，並記錄句子的順序索引（Index）。
* **實作程式碼範例**：
  ```python
  import re

  def split_to_sentences(text):
      """中文常見標點符號分句"""
      sentences = re.split(r'(。|！|？|；|\n)', text)
      result = []
      for i in range(0, len(sentences)-1, 2):
          combined = sentences[i] + sentences[i+1]
          if combined.strip():
              result.append(combined.strip())
      if len(sentences) % 2 != 0 and sentences[-1].strip():
          result.append(sentences[-1].strip())
      return result
  ```

### 5. Token 數量統計與動態截斷 (Token Count & Dynamic Truncation)
* **目的與優勢**：
  雙流程對比顯示，流程 A (Top-2) 檢索出上下文常達 2000-3000 字，這可能導致 LLM 生成延遲增加或 API 超載。我們需要一個安全閥，動態監控 Token 數量。
* **具體應用方式**：
  在 Prompt 組合前，調用 Token 計算機制，若超過最大限制，則動態縮減 Top-K 數量或截斷最末尾的文檔。
* **實作程式碼範例**：
  ```python
  # 使用 tiktoken 或 google-genai client 提供的計量接口
  def check_and_truncate_context(chunks, max_tokens=2500):
      # 對中文而言，字數（字元數）與 Token 數的估算比率大約是 1:1.5 到 1:2
      current_tokens = 0
      selected_chunks = []
      for chunk in chunks:
          chunk_tokens = len(chunk) * 1.5  # 估算值
          if current_tokens + chunk_tokens > max_tokens:
              break
          selected_chunks.append(chunk)
          current_tokens += chunk_tokens
      return selected_chunks
  ```

---

## 🚀 對於 v2.0 專案開發之整合策略

為達成 [PRD.md](file:///c:/Users/Kevin/Desktop/NEWFILE/NLP/FinalProject/PRD.md) 中關於 **「雙階段檢索架構 (Dense + Re-ranking)」** 以及 **「Web App 互動介面」** 的目標，這些建議將以以下形式嵌入專案：

1. **`clean_text` 的強化**：將語言偵測 `filter_chinese_posts` 嵌入 `update_pipeline.py` 腳本，從源頭過濾噪訊。
2. **重排序前的字數限制**：粗篩出的 Top-10 候選文檔，在進入 Cross-Encoder 重排序前，先利用 `split_text_with_overlap` 進行動態滑動視窗切分，確保 Cross-Encoder 比對的是語意高濃度的片段，而不是整篇冗長貼文。
3. **安全閥 (Safety Guardrail)**：在 Streamlit 呼叫 Gemini 前，執行 `check_and_truncate_context`，確保 Latency 控制在 PRD 要求的 **5-8 秒** 成功指標內。
