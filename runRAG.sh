#!/bin/bash

echo "==============================="
echo "Starting RAG server on port 8000"
echo "==============================="

python3 RAG/run.py --modelembedding "bge-m3" --kchunk 20 --sizechunk 5000
