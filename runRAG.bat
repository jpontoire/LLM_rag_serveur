@echo off

echo ===============================
echo Starting RAG server on port 8000
echo ===============================
python RAG/run.py --modelembedding "bge-m3" --kchunk 4 --sizechunk 4000
pause