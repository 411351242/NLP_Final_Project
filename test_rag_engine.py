import unittest
import pandas as pd
import numpy as np
from rag_engine import TextCleaner, VectorIndexer

class TestTextCleaner(unittest.TestCase):
    def test_clean_text_basic(self):
        text = "Hello @username, check this out: https://example.com! #finance (續"
        cleaned = TextCleaner.clean_text(text)
        self.assertEqual(cleaned, "Hello check this out:")

    def test_clean_text_chinese_english_spacing(self):
        text = "日圓Carry Trade套利交易"
        cleaned = TextCleaner.clean_text(text)
        self.assertEqual(cleaned, "日圓 Carry Trade 套利交易")

    def test_split_text_with_overlap(self):
        text = "abcdefghij"
        chunks = TextCleaner.split_text_with_overlap(text, chunk_size=5, overlap=2)
        self.assertGreater(len(chunks), 1)

    def test_check_and_truncate_context(self):
        contexts = ["short context", "another short context", "a very long context " * 100]
        # Char count * 1.5. If max_tokens is 100, first two should fit, third should be truncated
        truncated = TextCleaner.check_and_truncate_context(contexts, max_tokens=100)
        self.assertEqual(len(truncated), 2)

class TestVectorIndexer(unittest.TestCase):
    def test_vector_indexer_empty(self):
        indexer = VectorIndexer()
        self.assertEqual(len(indexer.documents), 0)
        results = indexer.search("query")
        self.assertEqual(results, [])

    def test_vector_indexer_simple(self):
        from unittest.mock import MagicMock
        indexer = VectorIndexer()
        indexer.model = MagicMock()
        indexer.model.encode.side_effect = lambda docs, **kwargs: (
            np.array([[1.0, 0.0, 0.0]]) if len(docs) == 1 else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        
        indexer.fit(["doc1", "doc2"], ["id1", "id2"])
        self.assertEqual(len(indexer.documents), 2)
        self.assertEqual(indexer.post_ids, ["id1", "id2"])
        
        results = indexer.search("query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['post_id'], 'id1')

if __name__ == "__main__":
    unittest.main()
