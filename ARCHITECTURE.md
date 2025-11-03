# FPT University RAG Chatbot - Architecture

## System Overview

This document describes the architecture of the production-ready RAG (Retrieval-Augmented Generation) chatbot for FPT University.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                  (FastAPI REST API)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Query Processing                        │
│         (Normalization, Tokenization, Embedding)             │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌─────────────────┐                   ┌─────────────────┐
│  BM25 (Sparse)  │                   │ FAISS (Dense)   │
│   rank_bm25     │                   │  Inner Product  │
│   Top-K=50      │                   │   Top-K=50      │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        │ Hybrid Fusion
                        │ (Normalized Score Merge)
                        ▼
            ┌────────────────────────┐
            │  Deduplicate & Merge   │
            │    Top-K=100           │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  Cross-Encoder Rerank  │
            │  BAAI/bge-reranker     │
            │     Top-K=8            │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  Context Builder       │
            │  (Format with Sources) │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │   Gemini LLM           │
            │   Answer Generation    │
            │   + Citations          │
            └────────────────────────┘
```

## Components

### 1. Data Ingestion Pipeline

#### 1.1 PDF Loader (`fptuchatbot/ingest/pdf_loader.py`)

**Purpose**: Extract text from Vietnamese PDFs with multiple strategies.

**Strategy Hierarchy**:
1. **Unstructured** (Primary)
   - `partition_pdf` with `strategy="hi_res"`
   - OCR with `languages=["vie"]`
   - Table structure detection
   - Best for: Complex layouts, scanned documents

2. **PyMuPDF** (Fallback)
   - Fast text extraction
   - Optional OCR with Tesseract
   - Footer removal using coordinate-based filtering
   - Best for: Simple text-based PDFs

**Features**:
- Auto-detection of scanned vs text-based PDFs
- Footer removal (configurable threshold: 70px from bottom)
- Page-level metadata preservation
- UTF-8 Vietnamese character handling

#### 1.2 Table Extractor (`fptuchatbot/ingest/table_extractor.py`)

**Purpose**: Extract structured tables from PDFs.

**Methods**:
- **Lattice**: For tables with visible borders
- **Stream**: For tables without borders (whitespace-based)
- Auto-fallback between methods

**Output**:
- Pandas DataFrame
- Markdown representation (for chunk embedding)
- Accuracy scores

#### 1.3 Document Chunker (`fptuchatbot/ingest/chunking.py`)

**Purpose**: Intelligent chunking with structure preservation.

**Semantic Chunking** (Default):
- Detects Vietnamese document structure:
  - `Chương I`, `Chương 1` (Chapters)
  - `Điều 1`, `Điều 2` (Articles)
  - `Khoản 1`, `Mục 1` (Sections)
  - `A.`, `B.`, `1.`, `2.` (List items)

- Preserves headers in chunks for context
- Paragraph-based splitting
- Overlap with intelligent boundary detection
- Token counting with tiktoken

**Simple Chunking** (Alternative):
- Sentence-based splitting
- Fixed overlap
- No structure awareness

**Configuration**:
- `chunk_size`: 800 characters (default)
- `chunk_overlap`: 200 characters
- `min_tokens`: 50 tokens minimum

#### 1.4 Embedder (`fptuchatbot/ingest/embedder.py`)

**Purpose**: Generate dense vector embeddings.

**Model**: `BAAI/bge-m3`
- Dimension: 1024
- Multilingual (excellent for Vietnamese)
- Pre-trained on diverse corpus
- Normalized embeddings for cosine similarity

**Fallback**: `intfloat/multilingual-e5-base`
- Dimension: 768
- Broader language support

**Features**:
- Batch processing (default: 32)
- GPU support (CUDA)
- File-based caching with hash keys
- Progress tracking

### 2. Indexing

#### 2.1 FAISS Index (`fptuchatbot/ingest/index_faiss.py`)

**Purpose**: Fast approximate nearest neighbor search.

**Index Types**:

1. **FlatIP** (Default, <100k vectors)
   - Exact search (100% recall)
   - Inner product similarity
   - Best for: Accuracy, smaller datasets
   - Memory: ~4-8GB for 10k vectors (1024-dim)

2. **HNSW** (For >100k vectors)
   - Approximate search (>95% recall)
   - Hierarchical Navigable Small World graph
   - Parameters:
     - `M=32`: Links per node
     - `efConstruction=200`: Build quality
     - `efSearch=64`: Search quality
   - Best for: Speed, large datasets

**Operations**:
- Add vectors with automatic normalization
- Top-K search with scores
- Save/load with pickle for chunk metadata

#### 2.2 BM25 Index (`fptuchatbot/ingest/bm25.py`)

**Purpose**: Sparse retrieval based on term frequency.

**Algorithm**: BM25 (Okapi BM25)
- Classic IR algorithm
- Parameters: k1=1.5, b=0.75 (default in rank_bm25)

**Tokenization**:
- **Simple** (Default): Whitespace splitting, lowercase
- **Underthesea** (Optional): Vietnamese word segmentation

**Features**:
- Vietnamese stopword removal (optional)
- Token length filtering
- Inverted index for fast lookup

### 3. Retrieval Pipeline

#### 3.1 Hybrid Retriever (`fptuchatbot/retrieval/hybrid.py`)

**Purpose**: Combine sparse (BM25) and dense (FAISS) retrieval.

**Algorithm**:

1. **Parallel Retrieval**:
   ```
   BM25:  query → tokenize → retrieve top-50
   FAISS: query → embed → retrieve top-50
   ```

2. **Score Normalization**:
   ```
   normalized_score = (score - min) / (max - min)
   ```
   Ensures scores from different retrievers are comparable.

3. **Fusion**:
   ```
   hybrid_score = α * BM25_norm + β * FAISS_norm
   ```
   Default: α=0.5, β=0.5 (equal weight)

4. **Deduplication**:
   - Merge by chunk_id or (source, page, text_hash)
   - Keep max score for duplicates

5. **Ranking**:
   - Sort by hybrid_score descending
   - Return top-K (default: 100 for reranking)

**Rationale**:
- BM25: Catches exact keyword matches, rare terms
- FAISS: Captures semantic similarity, paraphrases
- Combination: Higher recall than either alone

#### 3.2 Cross-Encoder Reranker (`fptuchatbot/retrieval/rerank.py`)

**Purpose**: Refine ranking with query-document interaction.

**Model**: `BAAI/bge-reranker-base`
- Bidirectional cross-encoder
- Jointly encodes query + document
- Higher quality than bi-encoder (embedder)
- Slower (not used for initial retrieval)

**Process**:
1. Take top-K from hybrid retrieval (e.g., 100)
2. Form query-document pairs
3. Batch predict relevance scores
4. Re-sort by rerank score
5. Return final top-K (e.g., 8)

**Trade-off**:
- Accuracy: ↑↑ (10-20% improvement typical)
- Latency: ↑ (~100-200ms for 100 pairs)

### 4. LLM Integration

#### 4.1 Gemini Client (`fptuchatbot/llm/gemini_client.py`)

**Purpose**: Generate natural language answers.

**Model**: `gemini-2.0-flash-exp`
- Fast inference (~1-2s)
- Good quality
- Vietnamese support
- Large context window (32k tokens)

**Prompt Structure**:
```
System: Bạn là trợ lý AI của Trường ĐH FPT...

Context:
[1] (Nguồn: file.pdf, Trang 5)
[Retrieved text chunk 1...]

[2] (Nguồn: file2.pdf, Trang 10)
[Retrieved text chunk 2...]

...

Query: [User question]

Instructions:
- Trả lời bằng tiếng Việt
- Chỉ dùng thông tin từ context
- Trích dẫn nguồn
- Ngắn gọn, chính xác

Answer:
```

**Features**:
- Context builder with citations
- Fallback error handling
- Temperature control
- Streaming support (future)

#### 4.2 Prompt Builder (`fptuchatbot/llm/prompts.py`)

**Templates**:
- `build_answer_prompt`: Main Q&A
- `build_query_rewrite_prompt`: Query expansion
- `build_system_instruction`: System role
- `build_followup_prompt`: Multi-turn chat

### 5. API Server

#### 5.1 FastAPI Application (`fptuchatbot/server/api.py`)

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/query` | POST | Full RAG query |
| `/search` | POST | Retrieval only (no LLM) |
| `/ingest` | POST | Trigger ingestion |
| `/reload` | POST | Reload indexes |

**Startup**:
- Load indexes from disk
- Initialize embedder, reranker, LLM client
- Warmup (optional)

**State Management**:
- Global instances (singleton pattern)
- Lazy loading for expensive models
- Graceful degradation if components fail

#### 5.2 Schemas (`fptuchatbot/server/schemas.py`)

**Pydantic Models**:
- Input validation
- Type safety
- Auto-generated OpenAPI docs
- Response formatting

### 6. Data Audit

#### 6.1 Data Auditor (`fptuchatbot/ingest/data_audit.py`)

**Purpose**: Quality assurance and coverage analysis.

**Manifest**:
- File metadata (size, pages, modification time)
- Topic classification (heuristic from filename)
- Language detection

**Quality Metrics**:
- Parse success rate
- OCR usage rate
- Table extraction count
- Errors and warnings

**Coverage Probe**:
- 40+ representative questions
- Measures recall@K for each question
- Identifies data gaps
- Outputs:
  - Coverage rate (% questions with results)
  - Unanswered questions (prioritize data collection)
  - Sample results per question

**Output**: `indexes/data_report.json`

## Data Flow

### Ingestion Flow

```
PDFs → PDF Loader → Pages
                      ↓
              Table Extractor → Merged Pages
                                      ↓
                               Chunker → Chunks
                                           ↓
                     ┌────────────────────┴────────────────────┐
                     ↓                                         ↓
               Embedder → Embeddings                   BM25 Builder
                     ↓                                         ↓
            FAISS Index Builder                         BM25 Index
                     ↓                                         ↓
                  Save                                      Save
```

### Query Flow

```
User Query → Normalize
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  BM25                FAISS
    ↓                   ↓
    └─────────┬─────────┘
              ↓
        Hybrid Merge
              ↓
          Reranker
              ↓
       Context Builder
              ↓
        Gemini LLM
              ↓
      Answer + Citations
```

## Configuration Management

**Environment Variables** (`.env`):
- Loaded with `pydantic-settings`
- Type validation
- Default values
- Hierarchical override: ENV > .env > defaults

**Settings Class** (`fptuchatbot/utils/config.py`):
- Singleton pattern with `@lru_cache`
- Ensures directories exist
- Path utilities (get_*_dir)

## Logging & Monitoring

**Logging** (`fptuchatbot/utils/logging.py`):
- Structured logging
- Console + file handlers
- Different formats (simple/detailed)
- UTF-8 encoding for Vietnamese

**Timing** (`fptuchatbot/utils/timing.py`):
- Context manager for timing blocks
- Decorator for function timing
- Human-readable formatting

**Metrics** (Future):
- Prometheus integration
- Latency histograms
- Error rates
- Cache hit rates

## Performance Optimization

### Indexing
- **Batch Embedding**: Process 32 chunks at once
- **Parallel PDF Loading**: Multi-threaded I/O
- **Incremental Indexing**: Add to existing index without rebuild

### Retrieval
- **FAISS HNSW**: Approximate search for large datasets
- **BM25 Pruning**: Pre-filter low IDF terms
- **Reranker Batching**: Process multiple pairs per inference

### Serving
- **Index Caching**: Load once, serve many
- **Embedding Caching**: File-based with hash keys
- **Async API**: Non-blocking I/O with FastAPI

## Error Handling

**Graceful Degradation**:
- PDF Loader: Unstructured → PyMuPDF → Skip file
- Table Extractor: Lattice → Stream → Skip tables
- Embedder: Primary model → Fallback model → Error
- LLM: Gemini → Error message (no fallback)

**Validation**:
- Input: Pydantic schemas
- File existence: Path checks
- Index consistency: Chunk count matching

## Security Considerations

- **API Keys**: Environment variables, never in code
- **Input Sanitization**: Pydantic validation
- **Logging**: No sensitive data in logs
- **CORS**: Configurable origins
- **Rate Limiting**: Future (not implemented)

## Scalability

**Current Limits** (Single Machine):
- PDFs: 100-1000 files
- Chunks: 10k-100k
- Queries: 10-100 QPS

**Scaling Strategies**:
1. **Horizontal** (Multiple Instances):
   - Stateless API servers
   - Shared index storage (NFS/S3)
   - Load balancer

2. **Vertical** (Bigger Machine):
   - GPU for embedding/reranking
   - More RAM for larger FAISS index
   - SSD for faster index loading

3. **Distributed Indexing**:
   - Shard FAISS index
   - Distribute BM25 across nodes
   - Merge results

## Testing Strategy

**Unit Tests**:
- Isolated component testing
- Mocked dependencies
- Fast execution (<1s per test)

**Integration Tests**:
- Full pipeline with real indexes
- Marked with `@pytest.mark.integration`
- Slower (<10s per test)

**Coverage Target**: >80%

## Future Enhancements

### Short-term
- [ ] Query rewriting with LLM
- [ ] Multi-turn conversation
- [ ] Response streaming
- [ ] Prometheus metrics

### Medium-term
- [ ] Fine-tuned embedder for FPT domain
- [ ] Hybrid search weight tuning
- [ ] Advanced PDF parsing (forms, stamps)
- [ ] Incremental indexing API

### Long-term
- [ ] Multi-modal support (images, videos)
- [ ] Distributed indexing
- [ ] Active learning (user feedback)
- [ ] Knowledge graph integration

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-30  
**Maintained By**: FPT University AI Team

