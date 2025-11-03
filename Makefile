.PHONY: help setup install format lint test audit ingest run clean

# Default target
help:
	@echo "FPT University RAG Chatbot - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Create conda environment and install dependencies"
	@echo "  install      - Install package in development mode"
	@echo "  format       - Format code with black and ruff"
	@echo "  lint         - Run linters (ruff)"
	@echo "  test         - Run tests with pytest"
	@echo "  audit        - Run data audit"
	@echo "  ingest       - Ingest PDFs and build indexes"
	@echo "  run          - Run FastAPI server"
	@echo "  clean        - Clean generated files"
	@echo ""

# Setup conda environment
setup:
	@echo "Creating conda environment 'rag311'..."
	conda create -n rag311 python=3.11 -y || true
	@echo "Installing dependencies..."
	conda run -n rag311 pip install -U pip setuptools wheel
	conda run -n rag311 pip install -r requirements.txt
	conda run -n rag311 pip install pytest pytest-cov pytest-asyncio pytest-mock black ruff
	@echo "Setup complete! Activate with: conda activate rag311"

# Install in development mode
install:
	pip install -e .

# Format code
format:
	@echo "Formatting code..."
	black fptuchatbot scripts tests
	ruff check --fix fptuchatbot scripts tests

# Lint code
lint:
	@echo "Running linters..."
	ruff check fptuchatbot scripts tests
	@echo "Linting complete!"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=fptuchatbot --cov-report=term-missing
	@echo "Tests complete!"

# Data audit
audit:
	@echo "Running data audit..."
	python scripts/ingest.py --audit-only
	@echo "Audit complete! Check indexes/data_report.json"

# Ingest data
ingest:
	@echo "Ingesting data and building indexes..."
	python scripts/ingest.py --run-audit
	@echo "Ingestion complete!"

# Reindex (force rebuild)
reindex:
	@echo "Rebuilding indexes..."
	python scripts/ingest.py --force
	@echo "Reindex complete!"

# Run server
run:
	@echo "Starting FastAPI server..."
	bash scripts/run_server.sh

# Run server with reload (development)
run-dev:
	@echo "Starting server in development mode..."
	RELOAD=true bash scripts/run_server.sh

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/
	@echo "Clean complete!"

# Search example
search:
	@echo "Example search (use: make search QUERY='your question')"
	python scripts/search.py -q "${QUERY}" --show-scores

# Full pipeline
all: format lint test ingest
	@echo "Full pipeline complete!"

