#!/bin/bash

echo "==============================="
echo "Starting RAG server on port 8000"
echo "==============================="

python3 RAG/run.py --modelembedding "bge-m3" --kchunk 4 --sizechunk 1000
