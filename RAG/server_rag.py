#server_rag.py
from fastapi import FastAPI
from pydantic import BaseModel
from init_rag import init_model_rag
import os
import time
from datetime import datetime

app = FastAPI()

class Query(BaseModel):
    prompt: str

# === Répertoires de logs ===
root_log = "LOGS"
dir_log = "SERVER"
log_dir = os.path.join(root_log, dir_log)
os.makedirs(log_dir, exist_ok=True)

# === Initialisation pipeline ===
k = int(os.getenv("K_CHUNK"))
size = int(os.getenv("SIZE_CHUNK"))
me = str(os.getenv("MODEL_EMBEDDING"))
print("[SERV-RAG] Initialisation SERVER RAG...")
t_init = time.time()
retriever, llm, custom_prompt, k_chunk, model_embedding = init_model_rag(data_dir="DATA", k_chunk=k, model_embedding=me, size_chunk=size)
exec_time_init = time.time() - t_init
print(f"[INIT] Chargement du modèle RAG en {exec_time_init:.2f} sec")

def log_request_txt(prompt: str, answer: str, exec_time: float, source_docs=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(log_dir, f"log_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Time : {timestamp}\n")
        f.write(f"Durée de la requete : {exec_time:.2f} sec\n")
        f.write(f"EMBEDDING : {model_embedding}\n")
        f.write(f"K_CHUNK : {k_chunk}\n")
        f.write(f"SIZE_CHUNK : {size}\n")
        f.write("\nPrompt :\n")
        f.write(prompt.strip() + "\n\n")
        f.write("Réponse :\n")
        f.write(answer.strip() + "\n")

        if source_docs:
            f.write("\n=== CHUNKS utilisés ===\n")
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get("source", "inconnu")
                extrait = doc.page_content.strip()
                f.write(f"\n[{i+1}] Source : {source}\n")
                f.write(f"Extrait : {extrait}...\n")

# === Endpoint FastAPI ===
@app.post("/query")
def ask(query: Query):
    t0 = time.time()
    
    # --- Mesure 1 : Embedding & Retrieval ---
    t1 = time.time()
    docs = retriever.invoke(query.prompt)
    exec_time_retrieval = time.time() - t1
    # ----------------------------------------
    
    docs_reversed = list(reversed(docs))

    context = "\n\n".join(doc.page_content for doc in docs_reversed)
    
    prompt = custom_prompt.format(context=context, question=query.prompt)

    # --- Mesure 2 : LLM Generation ---
    t2 = time.time()
    # Appel du modèle
    answer = llm.invoke(prompt)
    exec_time_llm = time.time() - t2
    # ----------------------------------

    exec_time_total = time.time() - t0
    
    # Affichage des temps détaillés
    print(f"[TIME] Requête traitée en {exec_time_total:.2f} sec")
    print(f"       -> Retrieval (Embedding + Search): {exec_time_retrieval:.2f} sec")
    print(f"       -> LLM Generation: {exec_time_llm:.2f} sec")

    # Mise à jour de la fonction de log et du retour
    log_request_txt(query.prompt, answer, exec_time_total, docs)
    return {
        "answer": answer,
        "execution_time_sec": round(exec_time_total, 2),
        "time_retrieval_sec": round(exec_time_retrieval, 2),
        "time_llm_sec": round(exec_time_llm, 2)
    }