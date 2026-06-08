"""Neural search with bi-encoder + cross-encoder."""
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from elasticsearch import Elasticsearch
from typing import List, Dict

class NeuralSearchEngine:
    def __init__(self, es_host: str = "localhost:9200"):
        self.bi_encoder   = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.es = Elasticsearch(es_host)

    def index_documents(self, docs: List[Dict], index: str = "search"):
        embeddings = self.bi_encoder.encode([d["text"] for d in docs], batch_size=64, show_progress_bar=True)
        for doc, emb in zip(docs, embeddings):
            self.es.index(index=index, body={**doc, "embedding": emb.tolist()})

    def hybrid_search(self, query: str, top_k: int = 10, index: str = "search") -> List[Dict]:
        query_emb = self.bi_encoder.encode(query).tolist()
        # Dense search
        dense_hits = self.es.search(index=index, body={"query": {"script_score": {
            "query": {"match_all": {}}, "script": {"source": "cosineSimilarity(params.emb, 'embedding') + 1.0",
            "params": {"emb": query_emb}}}}, "size": top_k * 3})
        # Sparse BM25
        sparse_hits = self.es.search(index=index, body={"query": {"match": {"text": query}}, "size": top_k * 3})
        # Merge with RRF
        scores = {}
        for rank, hit in enumerate(dense_hits["hits"]["hits"]): scores[hit["_id"]] = scores.get(hit["_id"], 0) + 1 / (60 + rank)
        for rank, hit in enumerate(sparse_hits["hits"]["hits"]): scores[hit["_id"]] = scores.get(hit["_id"], 0) + 1 / (60 + rank)
        top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k * 2]
        candidates = [{"id": h["_id"], "text": h["_source"]["text"]} for h in dense_hits["hits"]["hits"] if h["_id"] in top_ids]
        # Cross-encoder re-ranking
        pairs = [[query, c["text"]] for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
        return [{"rank": i+1, **doc, "score": float(score)} for i, (doc, score) in enumerate(reranked[:top_k])]
