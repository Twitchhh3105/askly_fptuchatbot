# FPT University RAG Chatbot

Production-ready Retrieval-Augmented Generation (RAG) system for FPT University Q&A with hybrid retrieval, cross-encoder reranking, and Google Gemini integration.

## 🎯 Features

- **Hybrid Retrieval**: Combines BM25 (sparse) and FAISS (dense) for optimal recall
- **Cross-Encoder Reranking**: Improves result relevance with `BAAI/bge-reranker-base`
- **Vietnamese Language Support**: Optimized for Vietnamese text with UTF-8 handling
- **PDF Processing**: Handles complex PDFs with OCR, table extraction (Camelot), and footer removal
- **Semantic Chunking**: Preserves document structure with header-aware chunking
- **Google Gemini Integration**: Powered by `gemini-2.0-flash-exp` for answer generation
- **FastAPI Server**: Production-ready REST API with async support
- **Comprehensive Testing**: Unit and integration tests with pytest
- **Data Audit Tools**: Quality metrics and coverage analysis

## 📋 Requirements

- Python 3.11
- Conda (for environment management)
- Linux/WSL2 (tested on Ubuntu)
- Google Gemini API key

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate conda environment
make setup
conda activate rag311

# Or manually:
conda create -n rag311 python=3.11 -y
conda activate rag311
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your Gemini API key
nano .env
# Set: GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Ingest Data

```bash
# Run data audit and ingestion
make ingest

# Or use script directly:
python scripts/ingest.py --run-audit

# Force rebuild if needed:
python scripts/ingest.py --force
```

This will:
- Analyze all PDFs in `data/` directory
- Generate data quality report
- Extract text with OCR support
- Create semantic chunks
- Build FAISS and BM25 indexes
- Save indexes to `indexes/` directory

### 4. Run Server

```bash
# Production mode
make run

# Development mode (with auto-reload)
make run-dev

# Or directly:
bash scripts/run_server.sh
```

Server will start at `http://localhost:8000`

### 5. Query

#### Via API:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Học phí ngành công nghệ thông tin là bao nhiêu?",
    "top_k": 8,
    "use_rerank": true
  }'
```

#### Via CLI:

```bash
python scripts/search.py -q "Điều kiện tốt nghiệp?" -k 5 --show-scores
```

## 📚 API Endpoints

### `POST /query`
Generate answer with RAG pipeline

**Request:**
```json
{
  "query": "string",
  "top_k": 8,
  "use_rerank": true,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "answer": "string",
  "citations": [
    {"source": "file.pdf", "page": "5", "score": "0.95"}
  ],
  "num_sources": 8,
  "retrieval_time_ms": 150.5,
  "generation_time_ms": 1200.3,
  "total_time_ms": 1350.8
}
```

### `POST /search`
Search without LLM (retrieval only)

### `POST /ingest`
Trigger ingestion (admin)

### `GET /health`
Health check

### `GET /`
API info

## 🏗️ Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ├────────────┐
       │            │
       ▼            ▼
  ┌─────────┐  ┌─────────┐
  │  BM25   │  │  FAISS  │
  │ (sparse)│  │ (dense) │
  └────┬────┘  └────┬────┘
       │            │
       └─────┬──────┘
             │ Hybrid Fusion
             ▼
      ┌──────────────┐
      │   Reranker   │
      │ Cross-Encoder│
      └──────┬───────┘
             │ Top-K
             ▼
      ┌──────────────┐
      │    Gemini    │
      │    Answer    │
      └──────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## 📊 Data Audit

Run comprehensive data analysis:

```bash
make audit
```

Generates `indexes/data_report.json` with:
- **Manifest**: File metadata, topics, sizes
- **Quality Metrics**: Parse success rate, OCR usage, table extraction
- **Coverage Probe**: 40+ test questions → recall metrics

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_bm25.py -v

# With coverage
pytest --cov=fptuchatbot --cov-report=html
```

## 🎨 Code Quality

```bash
# Format code
make format

# Run linter
make lint
```

## 📁 Project Structure

```
fptuchatbot/
├── fptuchatbot/           # Main package
│   ├── ingest/            # Data ingestion
│   │   ├── pdf_loader.py      # PDF extraction
│   │   ├── table_extractor.py # Camelot tables
│   │   ├── chunking.py        # Semantic chunking
│   │   ├── embedder.py        # bge-m3 embeddings
│   │   ├── index_faiss.py     # FAISS index
│   │   ├── bm25.py            # BM25 index
│   │   └── data_audit.py      # Quality analysis
│   ├── retrieval/         # Hybrid retrieval
│   │   ├── hybrid.py          # BM25 + FAISS fusion
│   │   └── rerank.py          # Cross-encoder
│   ├── llm/               # LLM integration
│   │   ├── gemini_client.py   # Gemini API
│   │   └── prompts.py         # Templates
│   ├── server/            # FastAPI
│   │   ├── api.py             # Endpoints
│   │   └── schemas.py         # Pydantic models
│   └── utils/             # Utilities
│       ├── config.py          # Settings
│       ├── logging.py         # Logging
│       └── timing.py          # Performance
├── scripts/               # CLI tools
│   ├── ingest.py          # Data ingestion
│   ├── search.py          # Search CLI
│   └── run_server.sh      # Server startup
├── tests/                 # Test suite
├── data/                  # Source PDFs
├── indexes/               # Generated indexes
├── logs/                  # Application logs
├── Makefile               # Build targets
├── pyproject.toml         # Package config
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## ⚙️ Configuration

Edit `.env` to customize:

### Models
- `EMBEDDING_MODEL`: Default `BAAI/bge-m3` (1024-dim, multilingual)
- `RERANKER_MODEL`: Default `BAAI/bge-reranker-base`
- `GEMINI_MODEL`: Default `gemini-2.0-flash-exp`

### Retrieval
- `TOP_K_BM25`: BM25 candidates (default: 50)
- `TOP_K_DENSE`: FAISS candidates (default: 50)
- `TOP_K_RERANK`: Rerank pool (default: 100)
- `TOP_K_FINAL`: Final results (default: 8)

### Chunking
- `CHUNK_SIZE`: Characters per chunk (default: 800)
- `CHUNK_OVERLAP`: Overlap size (default: 200)
- `USE_SEMANTIC_CHUNKING`: Preserve structure (default: true)

### PDF Processing
- `PDF_STRATEGY`: Unstructured strategy (default: hi_res)
- `OCR_LANGUAGES`: Tesseract langs (default: vie+eng)
- `USE_TABLE_EXTRACTION`: Enable Camelot (default: true)


## 🔒 Security

- API keys in `.env` (never committed)
- No logging of sensitive content
- CORS configured (adjust for production)
- Input validation with Pydantic

## 🐛 Troubleshooting

### "Indexes not loaded"
- Run `make ingest` first
- Check `indexes/` directory exists
- Verify no errors in logs

### "Gemini not configured"
- Set `GEMINI_API_KEY` in `.env`
- Restart server after changing `.env`

### OCR fails
- Install Tesseract: `sudo apt-get install tesseract-ocr tesseract-ocr-vie`
- Check `pytesseract` installation

### Camelot table extraction fails
- Install dependencies: `sudo apt-get install ghostscript python3-tk`
- Try alternative `PDF_STRATEGY=fast`

### Out of memory
- Reduce `BATCH_SIZE_EMBED` and `BATCH_SIZE_RERANK`
- Process PDFs in batches
- Use smaller embedding model

## 📝 License

MIT License - see LICENSE file

## 👥 Contributors

FPT University AI Team

## 📞 Support

For issues, create a GitHub issue or contact the development team.

---

**Built with ❤️ for FPT University students**

