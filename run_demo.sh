#!/bin/bash
# Quick demo script for FPT Chatbot

echo "🚀 FPT University RAG Chatbot - Quick Start"
echo "=========================================="
echo ""

# Activate conda
echo "📦 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate rag311

# Check if indexes exist
if [ ! -f "indexes/bm25_index.pkl" ] || [ ! -f "indexes/faiss_index.faiss" ]; then
    echo "⚠️  Indexes not found. Running ingestion first..."
    echo ""
    read -p "Do you want to run ingestion now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📚 Ingesting data (this may take 2-5 minutes)..."
        python scripts/ingest.py
        echo ""
    else
        echo "❌ Cannot continue without indexes. Exiting."
        exit 1
    fi
else
    echo "✅ Indexes found!"
fi

# Check Gemini API key
if grep -q "your_gemini_api_key_here" .env 2>/dev/null || ! grep -q "GEMINI_API_KEY=" .env 2>/dev/null; then
    echo "⚠️  GEMINI_API_KEY not configured!"
    echo "Please edit .env and set your Gemini API key"
    echo ""
fi

# Start server
echo ""
echo "🌐 Starting server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn fptuchatbot.server.api:app --host 0.0.0.0 --port 8000

