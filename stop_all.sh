#!/bin/bash

###############################################################################
# FPTU Chatbot - Stop All Services
# Dừng Backend API + Frontend UI
###############################################################################

# Màu sắc
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping FPTU Chatbot services...${NC}"
echo ""

# Stop processes on port 8000 (Backend)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${CYAN}Stopping Backend (port 8000)...${NC}"
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    echo -e "${GREEN}✅ Backend stopped${NC}"
else
    echo -e "${CYAN}Backend not running${NC}"
fi

# Stop processes on port 5173 (Frontend)
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${CYAN}Stopping Frontend (port 5173)...${NC}"
    lsof -ti:5173 | xargs kill -9 2>/dev/null
    echo -e "${GREEN}✅ Frontend stopped${NC}"
else
    echo -e "${CYAN}Frontend not running${NC}"
fi

# Clean up any uvicorn processes
pkill -f "uvicorn fptuchatbot.server.api:app" 2>/dev/null || true

# Clean up any npm/vite processes
pkill -f "vite" 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
