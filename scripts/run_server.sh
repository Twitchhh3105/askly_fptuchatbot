#!/bin/bash
# Run FastAPI server

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rag311

# Run server
cd "$(dirname "$0")/.."
uvicorn fptuchatbot.server.api:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8000} \
    --workers ${WORKERS:-1} \
    --log-level ${LOG_LEVEL:-info} \
    ${RELOAD:+--reload}

