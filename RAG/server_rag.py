from fastapi import FastAPI, Request
from pydantic import BaseModel
from init_rag import init_model_rag
import os
import time
import json
from datetime import datetime

MAX_HISTORY_TURNS = 6 

app = FastAPI()

class Query(BaseModel):
    prompt: str
    session_id: str = "default"

class ResetRequest(BaseModel):
    session_id: str = "default"

root_log = "LOGS"
dir_log = "SERVER"
log_dir = os.path.join(root_log, dir_log)
os.makedirs(log_dir, exist_ok=True)

k = int(os.getenv("K_CHUNK", 4))
size = int(os.getenv("SIZE_CHUNK", 2000))
me = str(os.getenv("MODEL_EMBEDDING", "bge-m3"))

print("[SERV-RAG] Initialisation SERVER RAG...")
t_init = time.time()

retriever, llm, custom_prompt, k_chunk, model_embedding = init_model_rag(
    data_dir="DATA",
    k_chunk=k,
    model_embedding=me,
    size_chunk=size
)

exec_time_init = time.time() - t_init
print(f"[INIT] Modèle RAG chargé en {exec_time_init:.2f} sec")

HISTORY_FILE = "server_memory.json" # Fichier de sauvegarde locale

def load_sessions():
    """Charge l'historique depuis le disque au démarrage."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[PERSISTENCE] Historique chargé ({len(data)} sessions).")
                return data
        except Exception as e:
            print(f"[PERSISTENCE] Erreur lecture fichier : {e}")
    return {}

def save_sessions():
    """Sauvegarde l'historique sur le disque."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(SESSIONS, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[PERSISTENCE] Erreur écriture fichier : {e}")

SESSIONS = load_sessions()

def get_formatted_history(session_id: str) -> str:
    if session_id not in SESSIONS:
        return ""
    
    history_list = SESSIONS[session_id]
    formatted_str = ""
    for user_msg, ai_msg in history_list:
        formatted_str += f"Utilisateur: {user_msg}\nAssistant: {ai_msg}\n"
    return formatted_str

def update_history(session_id: str, user_msg: str, ai_msg: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    
    SESSIONS[session_id].append((user_msg, ai_msg))
    
    if len(SESSIONS[session_id]) > MAX_HISTORY_TURNS:
        SESSIONS[session_id].pop(0)
        
    save_sessions()

def log_request_txt(prompt: str, answer: str, exec_time: float, session_id: str, source_docs=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_session_id = "".join([c for c in session_id if c.isalnum() or c in ('-', '_')]).rstrip()
    filename = os.path.join(log_dir, f"log_{timestamp}_{safe_session_id}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Time : {timestamp}\n")
        f.write(f"Session ID : {session_id}\n")
        f.write(f"Durée : {exec_time:.2f} sec\n")
        f.write("-" * 20 + "\n")
        f.write(prompt.strip() + "\n\n")
        f.write(answer.strip() + "\n")

@app.post("/query")
async def ask(query: Query):
    result = compute_rag(query.prompt, query.session_id)
    return result

def compute_rag(prompt: str, session_id: str):
    t0 = time.time()
    history_context = get_formatted_history(session_id)
    
    hist_len = len(SESSIONS.get(session_id, []))
    print(f"[{session_id}] Contexte : {hist_len} échanges précédents chargés.")

    docs = retriever.invoke(prompt)
    docs_reversed = list(reversed(docs))
    context_docs = "\n\n".join(doc.page_content for doc in docs_reversed)

    prompt_to_llm = custom_prompt.format(
        context=context_docs,
        question=prompt,
        history=history_context,
    )

    answer = llm.invoke(prompt_to_llm)
    exec_time_total = time.time() - t0

    log_request_txt(prompt, answer, exec_time_total, session_id, docs)
    update_history(session_id, prompt, answer)

    return {
        "answer": answer,
        "execution_time_sec": round(exec_time_total, 2),
        "session_id": session_id,
        "history_depth": hist_len
    }

@app.post("/reset")
async def reset_history(request: ResetRequest):
    session_id = request.session_id
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        save_sessions()
        msg = f"Historique '{session_id}' effacé."
    else:
        msg = f"Déjà vide."

    return {"status": "success", "message": msg}