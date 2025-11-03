#!/bin/bash

###############################################################################
# FPTU Chatbot - Start All Services
# Khởi động Backend API + Frontend UI cùng lúc
###############################################################################

set -e  # Exit on error

# Màu sắc
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_DIR/streamlit_app/front-end"

# PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

###############################################################################
# Cleanup function
###############################################################################
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping services...${NC}"
    
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${CYAN}Stopping Backend (PID: $BACKEND_PID)...${NC}"
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo -e "${CYAN}Stopping Frontend (PID: $FRONTEND_PID)...${NC}"
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Kill any remaining processes on ports 8000 and 5173
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

###############################################################################
# Banner
###############################################################################
echo -e "${MAGENTA}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🤖  FPTU CHATBOT - RAG SYSTEM  🤖                 ║
║                                                           ║
║     Retrieval-Augmented Generation Chatbot               ║
║     for FPT University                                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

###############################################################################
# Check prerequisites
###############################################################################
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

# Check conda environment
if ! conda env list | grep -q "rag311"; then
    echo -e "${RED}❌ Conda environment 'rag311' not found${NC}"
    echo -e "${YELLOW}Please run: make setup${NC}"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    echo -e "${YELLOW}Please install Node.js: https://nodejs.org/${NC}"
    exit 1
fi

# Check indexes
if [ ! -f "$PROJECT_DIR/indexes/faiss_index.faiss" ] || [ ! -f "$PROJECT_DIR/indexes/bm25_index.pkl" ]; then
    echo -e "${RED}❌ Indexes not found${NC}"
    echo -e "${YELLOW}Please run: python scripts/ingest.py --run-audit${NC}"
    exit 1
fi

# Check if ports are available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 8000 already in use, killing process...${NC}"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 5173 already in use, killing process...${NC}"
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

###############################################################################
# Install Frontend dependencies (if needed)
###############################################################################
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing Frontend dependencies...${NC}"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install Frontend dependencies${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
    echo ""
fi

###############################################################################
# Start Backend API
###############################################################################
echo -e "${BLUE}🚀 Starting Backend API...${NC}"

cd "$PROJECT_DIR"

# Activate conda and run backend in background
eval "$(conda shell.bash hook)"
conda activate rag311

# Start backend and capture PID
nohup uvicorn fptuchatbot.server.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > logs/backend.log 2>&1 &

BACKEND_PID=$!

echo -e "${GREEN}✅ Backend starting (PID: $BACKEND_PID)${NC}"
echo -e "${CYAN}   Log: logs/backend.log${NC}"

# Wait for backend to be ready
echo -n -e "${YELLOW}⏳ Waiting for Backend to be ready"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Check if backend is actually running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    echo -e "${YELLOW}Check logs: tail -f logs/backend.log${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}✅ Backend ready at http://localhost:8000${NC}"
echo ""

###############################################################################
# Start Frontend UI
###############################################################################
echo -e "${BLUE}🚀 Starting Frontend UI...${NC}"

cd "$FRONTEND_DIR"

# Start frontend in background
nohup npm run dev > "$PROJECT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo -e "${GREEN}✅ Frontend starting (PID: $FRONTEND_PID)${NC}"
echo -e "${CYAN}   Log: logs/frontend.log${NC}"

# Wait for frontend to be ready
echo -n -e "${YELLOW}⏳ Waiting for Frontend to be ready"
for i in {1..30}; do
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

###############################################################################
# Success message
###############################################################################
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                 🎉  ALL SERVICES RUNNING  🎉${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}📍 Backend API:${NC}      http://localhost:8000"
echo -e "${CYAN}📍 API Docs:${NC}         http://localhost:8000/docs"
echo -e "${CYAN}📍 Frontend UI:${NC}      http://localhost:5173"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo -e "   • Open ${CYAN}http://localhost:5173${NC} in your browser"
echo -e "   • Press ${RED}Ctrl+C${NC} to stop all services"
echo -e "   • View logs: ${CYAN}tail -f logs/backend.log${NC} or ${CYAN}logs/frontend.log${NC}"
echo ""
echo -e "${MAGENTA}🤖 Ready to chat! Ask me anything about FPT University!${NC}"
echo ""

# Open browser automatically (optional)
if command -v xdg-open &> /dev/null; then
    echo -e "${CYAN}🌐 Opening browser...${NC}"
    xdg-open http://localhost:5173 &
elif command -v open &> /dev/null; then
    echo -e "${CYAN}🌐 Opening browser...${NC}"
    open http://localhost:5173 &
fi

echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services...${NC}"
echo ""

# Wait for user interrupt
wait
