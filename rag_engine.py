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
    @staticmethod
    def clean_text(text):
        """
        Cleans raw social media text data by:
        1. Checking string type.
        2. Replacing all newlines (\n, \r, \\n, /n) with spaces to avoid breaking semantic context.
        3. Removing Unicode control chars and object replacements (\ufffc / ￼ and \u200b).
        4. Removing URLs and hyperlinks.
        5. Removing hashtags (#hashtag) and mentions (@username).
        6. Stripping trailing/internal platform noise ('(續', '（續', 'Read more', 'Reply to make_investment_easy...').
        7. Adding spacing between Chinese characters and English letters/digits.
        8. Collapsing multiple spaces into a single space.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Newlines to single space
        cleaned = re.sub(r'(?:/n|\\\\n|\\n|\\r)+', ' ', text)
        
        # 2. Unicode object replacements and zero-width spaces
        cleaned = cleaned.replace('\ufffc', '').replace('￼', '').replace('\u200b', '')
        
        # 3. URLs
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
        
        # 4. Social media tags
        cleaned = re.sub(r'#\S+', '', cleaned)
        cleaned = re.sub(r'@\S+', '', cleaned)
        
        # 5. Platforms suffixes and trailing tags
        cleaned = re.sub(r'(?i)reply\s+to\s+make_investment_easy\.*', '', cleaned)
        cleaned = re.sub(r'\(續\s*$', '', cleaned)
        cleaned = re.sub(r'（續\s*$', '', cleaned)
        cleaned = re.sub(r'Read more\s*$', '', cleaned)
        cleaned = re.sub(r'\(續\)?', '', cleaned)
        cleaned = re.sub(r'（續）?', '', cleaned)
        cleaned = re.sub(r'Read more', '', cleaned)
        
        # 6. Spacing between Chinese and English/number chars
        cleaned = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', cleaned)
        
        # 7. Collapsing multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()

    @staticmethod
    def split_text_with_overlap(text, chunk_size=600, overlap=150):
        """
        Splits a text string into overlapping chunks for better semantic retrieval and context control.
        """
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    @staticmethod
    def check_and_truncate_context(contexts, max_tokens=2500):
        """
        Filters a list of retrieved context chunks to make sure estimated tokens (char count * 1.5)
        does not exceed a safety threshold.
        """
        current_tokens = 0
        selected = []
        for ctx in contexts:
            # Estimate token count (chars * 1.5 represents reasonable ratio for mixed Chinese/English text)
            estimated_tokens = len(ctx) * 1.5
            if current_tokens + estimated_tokens > max_tokens:
                break
            selected.append(ctx)
            current_tokens += estimated_tokens
        return selected


class VectorIndexer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initializes the dense Vector Indexer using a multilingual SentenceTransformer model.
        """
        self.model_name = model_name
        self.model = None  # Lazy load the model
        self.embeddings = None
        self.documents = []
        self.post_ids = []

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model' in state:
            del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None  # Re-initialized lazily

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def fit(self, documents, post_ids=None):
        """
        Fits the indexer with documents and computes their dense vectors.
        """
        self._load_model()
        self.documents = list(documents)
        self.post_ids = list(post_ids) if post_ids is not None else [f"Doc_{i+1}" for i in range(len(documents))]
        
        if len(self.documents) > 0:
            self.embeddings = self.model.encode(self.documents, show_progress_bar=False)
        else:
            self.embeddings = np.array([])
        
    def search(self, query, top_k=10):
        """
        Performs first-stage dense retrieval using cosine similarity.
        """
        if len(self.documents) == 0:
            return []
        
        self._load_model()
        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Sort in descending order of similarity
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
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the Cross-Encoder model for high-precision second-stage re-ranking.
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidate_results, top_k=3):
        """
        Reranks the candidates based on query-document cross-attention scoring.
        """
        if not candidate_results:
            return []
        
        # Prepare pairs: (query, document)
        pairs = [(query, res['document']) for res in candidate_results]
        scores = self.model.predict(pairs)
        
        # Update scores in the candidate dictionaries
        for i, score in enumerate(scores):
            candidate_results[i]['rerank_score'] = float(score)
            
        # Sort by rerank score descending
        reranked = sorted(candidate_results, key=lambda x: x['rerank_score'], reverse=True)
        
        actual_top_k = min(top_k, len(reranked))
        return reranked[:actual_top_k]


class RateLimiter:
    LIMIT_FILE = "rate_limit_log.json"

    @classmethod
    def check_and_log(cls, max_rpm=10, max_rpd=200):
        now = time.time()
        
        # Load existing logs
        logs = []
        if os.path.exists(cls.LIMIT_FILE):
            try:
                with open(cls.LIMIT_FILE, "r") as f:
                    logs = json.load(f)
            except:
                pass
        
        # Filter logs to keep only the last 24 hours (86400 seconds)
        one_day_ago = now - 86400
        logs = [t for t in logs if t > one_day_ago]
        
        # Check RPD (Requests Per Day)
        if len(logs) >= max_rpd:
            raise RuntimeError(f"已達到單日 API 限制上限（RPD <= {max_rpd}），請稍後再試。")
            
        # Check RPM (Requests Per Minute)
        one_minute_ago = now - 60
        rpm_logs = [t for t in logs if t > one_minute_ago]
        if len(rpm_logs) >= max_rpm:
            # Calculate sleep time (spacing)
            oldest_rpm = rpm_logs[0]
            sleep_time = 60 - (now - oldest_rpm)
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()  # Update current timestamp
        
        # Log the current request
        logs.append(now)
        try:
            with open(cls.LIMIT_FILE, "w") as f:
                json.dump(logs, f)
        except:
            pass

class GeminiGenerator:
    def __init__(self, api_key=None, model_name='gemini-3.1-flash-lite-preview'):
        """
        Manages Generative LLM QA using Google Gemini API.
        """
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query, retrieved_docs, temperature=0.2):
        """
        Sends the query and retrieved context to Gemini and returns the generated response.
        """
        # Apply Rate Limiter before calling the Gemini API
        try:
            RateLimiter.check_and_log(max_rpm=10, max_rpd=200)
        except Exception as e:
            return f"⚠️ **安全防護（速率限制觸發）**：{str(e)}\n\n在展示期間為確保 API 金鑰不爆炸，系統設有 RPM <= 10 與 RPD <= 200 的慢速模式保護機制。"

        # Formulate context text
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
