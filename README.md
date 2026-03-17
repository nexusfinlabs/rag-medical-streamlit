# Agentic RAG — Medical Imaging (CT Lung Cancer)

**Hybrid RAG system** for 14,377 CT chest slices from TCIA CMB-LCA dataset.
Demonstrates **Senior AI Platform Backend Engineer** patterns.

## Architecture

```
INGESTION PIPELINE:
  DICOM (.dcm) → PNG slices → CLIP embeddings (512d)
  DICOM metadata → text → MiniLM embeddings (384d)
  Image ↔ Text → pairs.jsonl (joined index)

QUERY PIPELINE (Hybrid RAG):
  Query ──► BM25 Search (keyword)      ──► Top K₁
        ──► Vector Search (semantic)    ──► Top K₂
        ──► RRF Fusion                  ──► Merged Top N
        ──► Cross-Encoder Reranker      ──► Final Top M
        ──► LLM (Ollama/LLaMA)          ──► Answer + Sources

AGENTIC LAYER (LangGraph):
  classify_query → hybrid_search → evaluate_results
                                        │
                        ┌────────────────┤
                        ▼                ▼
                   need_more        sufficient
                        │                │
                        ▼                ▼
                  refine_query    generate_answer → END
                        │
                        └──► hybrid_search (loop, max 2)
```

## Stack

| Component | Library | Purpose |
|---|---|---|
| Image Embeddings | CLIP ViT-B/32 | Visual similarity |
| Text Embeddings | all-MiniLM-L6-v2 | Semantic search |
| BM25 | rank_bm25 | Keyword search |
| Reranker | cross-encoder/ms-marco-MiniLM | Result refinement |
| NLP | SpaCy + NLTK | Entity extraction, preprocessing |
| Agent | LangGraph | Multi-step reasoning |
| LLM | Ollama (LLaMA 3) | Answer generation |
| Vector DB | FAISS / NumPy | Vector storage |
| Demo | Streamlit | Interactive UI |

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run pipeline (if data exists)
bash run_pipeline.sh

# 3. Hybrid Search
python -m rag.hybrid_search --query "lung nodule SIEMENS 1.25mm" --k 10 --mode hybrid
python -m rag.hybrid_search --query "NER1006" --k 5 --mode bm25
python -m rag.hybrid_search --query "thin slice CT" --k 8 --rerank

# 4. Agentic RAG
python -m graph.workflow --question "Compare GE vs SIEMENS protocols"

# 5. Demo UI
streamlit run app.py
```

## Why Hybrid Search?

| Query Type | BM25 | Vector | Hybrid |
|---|---|---|---|
| Exact codes: `NER1006` | ✅ Best | ❌ Misses | ✅ Good |
| Semantic: `thin lung scans` | ❌ Misses | ✅ Best | ✅ Good |
| Mixed: `SIEMENS 1.25mm lung` | ⚠️ Partial | ⚠️ Partial | ✅ Best |

## Files

```
rag/
├── hybrid_search.py     # BM25 + Vector + RRF + Reranker
├── preprocessor.py      # SpaCy/NLTK medical NLP
├── retrieve.py          # Original vector retrieval
├── ask_ollama.py        # Original LLM query
├── retrieve_numpy.py    # NumPy-based retrieval
└── caption_*.py         # VLM captioning

graph/
└── workflow.py          # LangGraph multi-step agent

pipelines/
├── dicom/               # DICOM → PNG + metadata
├── embeddings/          # CLIP + MiniLM embeddings
└── ingest/              # Join image ↔ text

app.py                   # Streamlit demo
run_pipeline.sh          # Full pipeline runner
```
