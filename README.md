# 🔎 Neural Search Engine

[![QPS](https://img.shields.io/badge/QPS-28K-blue)](.) [![Relevance](https://img.shields.io/badge/NDCG%4010-0.847-green)](.) [![Languages](https://img.shields.io/badge/Languages-8-orange)](.)

> **Production neural search** with bi-encoder retrieval + cross-encoder re-ranking. **28K QPS**, **NDCG@10: 0.847** (vs 0.612 BM25 baseline). Hybrid search with query expansion and typo correction.

## 🏗️ Search Architecture
```
Query → Spell Correction → Query Expansion (LLM)
     → [Dense] Bi-encoder embedding → ANN search (HNSW)
     → [Sparse] BM25 → Elasticsearch
     → RRF Fusion → Cross-encoder Re-ranking → Results
```

## 📊 Ablation Study
| System | NDCG@10 | Latency |
|--------|---------|---------|
| BM25 only | 0.612 | 12ms |
| Dense only | 0.741 | 45ms |
| Hybrid (no re-rank) | 0.793 | 58ms |
| **Full system** | **0.847** | **84ms** |
