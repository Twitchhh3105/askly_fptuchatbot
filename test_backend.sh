#!/bin/bash

# Test Backend API

echo "🧪 Testing Backend API..."
echo ""

# Test 1: Health check
echo "1️⃣ Testing /health endpoint..."
HEALTH=$(curl -s http://localhost:8000/health)
if [ $? -eq 0 ]; then
    echo "✅ Backend is running"
    echo "$HEALTH" | jq .
else
    echo "❌ Backend is NOT running"
    exit 1
fi

echo ""

# Test 2: Root endpoint
echo "2️⃣ Testing / endpoint..."
ROOT=$(curl -s http://localhost:8000/)
if [ $? -eq 0 ]; then
    echo "✅ Root endpoint OK"
    echo "$ROOT" | jq .
else
    echo "❌ Root endpoint failed"
fi

echo ""

# Test 3: Search endpoint (no LLM)
echo "3️⃣ Testing /search endpoint..."
SEARCH=$(curl -s -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Học phí CNTT?", "top_k": 3}')

if [ $? -eq 0 ]; then
    echo "✅ Search endpoint OK"
    echo "$SEARCH" | jq .results[0].source 2>/dev/null || echo "$SEARCH"
else
    echo "❌ Search endpoint failed"
fi

echo ""

# Test 4: Query endpoint (with LLM)
echo "4️⃣ Testing /query endpoint (with LLM)..."
QUERY=$(curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Học phí CNTT là bao nhiêu?", "top_k": 3, "use_rerank": true}')

if [ $? -eq 0 ]; then
    echo "✅ Query endpoint OK"
    echo "$QUERY" | jq -r '.answer' 2>/dev/null || echo "$QUERY"
else
    echo "❌ Query endpoint failed"
    echo "$QUERY"
fi

echo ""
echo "✅ Backend test complete!"
