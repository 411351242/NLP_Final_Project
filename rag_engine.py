import re
import numpy as np
import pandas as pd
import os
import json
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


class TextCleaner:
    """
    社群文本清洗工具類別
    負責處理換行符號、特殊符號、網址、社群標籤與中英文間距標準化。
    """

    @staticmethod
    def clean_thread_item(text: str) -> str:
        """
        清理單則串文的基本格式與雜訊：
        1. 移除 Unicode 物件替換字元 (\ufffc, ￼) 與零寬/控制字元。
        2. 移除結尾的 (續 / （續 接續標記。
        3. 壓縮連續 3 個以上的換行符號為雙換行。
        """
        if not isinstance(text, str):
            return ""
        text_cleaned = re.sub(r'[\ufffc\ufffd\u200b-\u200f\ufeff￼]', '', text)
        text_cleaned = re.sub(r'[\(（\s]*續[\s\)*）]*$', '', text_cleaned)
        text_cleaned = re.sub(r'\n{3,}', '\n\n', text_cleaned)
        return text_cleaned.strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗原始文本資料：
        1. 檢查輸入型態是否為字串。
        2. 將各式換行符號替換為單一空格，維持語意上下文連貫。
        3. 移除 Unicode 控制字元與物件替換符號 (如 \\ufffc, \\u200b)。
        4. 移除 HTTP/HTTPS 網址與 www 連結。
        5. 移除社群標籤 (#hashtag) 與使用者標記 (@username)。
        6. 移除平台尾端標記 (如 '(續', 'Read more', 'Reply to ...')。
        7. 於中文字元與英數字元之間補齊空格，提高斷詞與檢索準確度。
        8. 將連續多個空格合併為單一空格並去除首尾空白。
        """
        if not isinstance(text, str):
            return ""

        # 1. 換行符號轉為單一空格
        cleaned = re.sub(r'(?:/n|\\\\n|\\n|\\r)+', ' ', text)

        # 2. 移除 Unicode 特殊字元
        cleaned = cleaned.replace('\ufffc', '').replace('￼', '').replace('\u200b', '')

        # 3. 移除網址
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)

        # 4. 移除社群標籤與標記
        cleaned = re.sub(r'#\S+', '', cleaned)
        cleaned = re.sub(r'@\S+', '', cleaned)

        # 5. 移除平台後綴與尾端噪訊
        cleaned = re.sub(r'(?i)reply\s+to\s+make_investment_easy\.*', '', cleaned)
        cleaned = re.sub(r'\(續\s*$', '', cleaned)
        cleaned = re.sub(r'（續\s*$', '', cleaned)
        cleaned = re.sub(r'Read more\s*$', '', cleaned)
        cleaned = re.sub(r'\(續\)?', '', cleaned)
        cleaned = re.sub(r'（續）?', '', cleaned)
        cleaned = re.sub(r'Read more', '', cleaned)

        # 6. 中英文與數字間距標準化
        cleaned = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', cleaned)

        # 7. 合併連續空格
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()


class VectorIndexer:
    """
    第一階段：密集向量索引器 (Dense Vector Indexer)
    使用 Sentence-Transformer 模型將文本轉換為 384 維語意向量，
    並使用餘弦相似度 (Cosine Similarity) 進行第一階段候選文章召回。
    """

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = None  # 延遲載入 (Lazy Load)，以利 pickle 序列化儲存
        self.embeddings = None
        self.documents = []
        self.post_ids = []

    def __getstate__(self):
        """序列化時排除 PyTorch 模型物件，大幅縮減 pickle 檔案體積"""
        state = self.__dict__.copy()
        if 'model' in state:
            del state['model']
        return state

    def __setstate__(self, state):
        """反序列化載入時恢復屬性，並將 model 重設為 None 待需要時載入"""
        self.__dict__.update(state)
        self.model = None

    def _load_model(self):
        """在首次進行向量編碼或搜尋時載入 SentenceTransformer 模型"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def fit(self, documents: list, post_ids: list = None):
        """
        將知識庫文本轉換為向量並建立索引。
        :param documents: 清洗後的文本列表
        :param post_ids: 對應的貼文編號列表
        """
        self._load_model()
        self.documents = list(documents)
        self.post_ids = list(post_ids) if post_ids is not None else [f"Doc_{i+1}" for i in range(len(documents))]

        if len(self.documents) > 0:
            self.embeddings = self.model.encode(self.documents, show_progress_bar=False)
        else:
            self.embeddings = np.array([])

    def search(self, query: str, top_k: int = 10) -> list:
        """
        第一階段初篩：依餘弦相似度搜尋最相關的 Top-K 篇候選貼文。
        :param query: 使用者提問字串
        :param top_k: 召回的候選篇數 (預設 10 篇)
        :return: 包含貼文 ID、文本內容與相似度分數的字典列表
        """
        if len(self.documents) == 0:
            return []

        self._load_model()
        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]

        actual_top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:actual_top_k]

        results = []
        for idx in top_indices:
            results.append({
                'post_id': self.post_ids[idx],
                'document': self.documents[idx],
                'score': float(similarities[idx]),
                'index': int(idx)
            })
        return results


class CrossEncoderReranker:
    """
    第二階段：深度重排序器 (Cross-Encoder Reranker)
    使用交叉注意力模型對 (問題, 候選文本) 進行一對一關聯度評分，精選最終送入 LLM 的貼文。
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_results: list, top_k: int = 3) -> list:
        """
        對第一階段召回的候選貼文進行重排序。
        :param query: 使用者提問
        :param candidate_results: 第一階段 VectorIndexer 輸出的候選列表
        :param top_k: 重排序後保留的核心貼文數量 (預設 3 篇)
        :return: 依 rerank_score 降序排列的貼文列表
        """
        if not candidate_results:
            return []

        pairs = [(query, res['document']) for res in candidate_results]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidate_results[i]['rerank_score'] = float(score)

        reranked = sorted(candidate_results, key=lambda x: x['rerank_score'], reverse=True)
        actual_top_k = min(top_k, len(reranked))
        return reranked[:actual_top_k]


class RateLimiter:
    """
    API 請求頻率限制器
    記錄呼叫時間戳記，控制每分鐘請求數 (RPM) 與每日請求數 (RPD)。
    """
    LIMIT_FILE = "rate_limit_log.json"

    @classmethod
    def check_and_log(cls, max_rpm: int = 10, max_rpd: int = 200):
        """
        檢查是否超出 API 呼叫頻率限制。
        若達每分鐘上限則自動 sleep 等待，若達每日上限則丟出例外中斷。
        :param max_rpm: 每分鐘最大允許請求數
        :param max_rpd: 每日最大允許請求數
        """
        now = time.time()

        logs = []
        if os.path.exists(cls.LIMIT_FILE):
            try:
                with open(cls.LIMIT_FILE, "r") as f:
                    logs = json.load(f)
            except Exception:
                pass

        # 只保留 24 小時內的紀錄
        one_day_ago = now - 86400
        logs = [t for t in logs if t > one_day_ago]

        # 每日請求數 (RPD) 檢查
        if len(logs) >= max_rpd:
            raise RuntimeError(f"已達到單日 API 限制上限 (RPD <= {max_rpd})，請明日再試。")

        # 每分鐘請求數 (RPM) 檢查
        one_minute_ago = now - 60
        rpm_logs = [t for t in logs if t > one_minute_ago]
        if len(rpm_logs) >= max_rpm:
            oldest_rpm = rpm_logs[0]
            sleep_time = 60 - (now - oldest_rpm)
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()

        logs.append(now)
        try:
            with open(cls.LIMIT_FILE, "w") as f:
                json.dump(logs, f)
        except Exception:
            pass


class GeminiGenerator:
    """
    LLM 生成器
    負責組裝檢索到的上下文與提問，呼叫 Google Gemini API 產生繁體中文回覆。
    """

    def __init__(self, api_key: str = None, model_name: str = 'gemini-3.1-flash-lite-preview'):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query: str, retrieved_docs: list, temperature: float = 0.2) -> str:
        """
        組裝 Prompt 並呼叫 Gemini 生成回答。
        :param query: 使用者提問
        :param retrieved_docs: 經重排序篩選出的核心貼文列表
        :param temperature: 生成溫度 (0.0~1.0)，越低表示越忠於原文
        :return: 模型回覆文字
        """
        try:
            RateLimiter.check_and_log(max_rpm=10, max_rpd=200)
        except Exception as e:
            return f"**速率限制提示**：{str(e)}\n\n為避免 API 請求超出配額，系統設有 RPM <= 10 與 RPD <= 200 的頻率保護。"

        # 組裝參考貼文內容
        context_str = ""
        for i, res in enumerate(retrieved_docs):
            score_type = 'rerank_score' if 'rerank_score' in res else 'score'
            score_val = res[score_type]
            context_str += f"[參考來源 {i+1} - {res['post_id']} (相關性評分: {score_val:.4f})]:\n{res['document']}\n---\n"

        prompt = f"""
你是一位專業的個人財經與職涯智庫助理。你的回答必須基於以下提供的「作者貼文內容」。
請以原作者（一位經驗豐富的投資銀行交易員、思維清晰的理性投資人）的風格口吻進行回答，並且簡要說明貼文邏輯。

【約束條件】：
1. 你的回答必須使用「繁體中文 (Traditional Chinese)」。
2. 必須只根據下方【參考貼文】的內容回答。如果參考貼文中完全沒有提到相關資訊，請坦白回答「抱歉，在作者的貼文資料庫中沒有找到相關的論述」。絕對不要胡亂編造或使用外部知識進行幻想 (Hallucination)。
3. 回答應條理分明、分點敘述，適當加入原作者常用的思維邏輯（例如：強調風險溢酬、資產負債表伸縮、或時間價值）。

【參考貼文】：
{context_str}

【使用者問題】：
{query}

請生成回答：
"""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )
        return response.text
