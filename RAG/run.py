import argparse
import uvicorn
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kchunk", type=int, default=4, help="K_CHUNK")
    parser.add_argument("--port", type=int, default=8000, help="PORT")
    parser.add_argument("--modelembedding", type=str, default="bge-m3", help="Model EMBEDDING (ex: bge-m3 for Ollama)")
    parser.add_argument("--sizechunk", type=int, default=2000, help="SIZE_CHUNK")
    args = parser.parse_args()

    host = "127.0.0.1"
    port = args.port

    # Variables d’environnement
    os.environ["K_CHUNK"] = str(args.kchunk)
    os.environ["SIZE_CHUNK"] = str(args.sizechunk)
    os.environ["MODEL_EMBEDDING"] = str(args.modelembedding)
    os.environ["EMBEDDING_PROVIDER"] = "ollama"

    print(f"[SERVER] START {host}:{port}")
    print(f"[SERVER] RAG options :")
    print(f"       K_CHUNK={args.kchunk}")
    print(f"       SIZE_CHUNK={args.sizechunk}")
    print(f"       EMBED={args.modelembedding} (via Ollama)")
    uvicorn.run(
    "server_rag:app",
    host=host,
    port=port,
    reload=False,
    workers=1,
    loop="asyncio"
)

if __name__ == "__main__":
    main()
