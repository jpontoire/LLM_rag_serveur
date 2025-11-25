from fastapi import FastAPI, Request
from pydantic import BaseModel
from init_rag import init_model_rag  # Assurez-vous que votre fichier init s'appelle bien init_rag.py
import os
import time
from datetime import datetime

MAX_HISTORY_TURNS = 6 
SESSIONS = {}

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

def get_formatted_history(session_id: str) -> str:
    """Récupère et formate l'historique pour une session donnée en string."""
    if session_id not in SESSIONS:
        return ""
    
    history_list = SESSIONS[session_id]
    formatted_str = ""
    for user_msg, ai_msg in history_list:
        formatted_str += f"Utilisateur: {user_msg}\nAssistant: {ai_msg}\n"
    return formatted_str

def update_history(session_id: str, user_msg: str, ai_msg: str):
    """Ajoute un échange à la session et supprime les vieux messages."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    
    SESSIONS[session_id].append((user_msg, ai_msg))
    
    if len(SESSIONS[session_id]) > MAX_HISTORY_TURNS:
        SESSIONS[session_id].pop(0)

def log_request_txt(prompt: str, answer: str, exec_time: float, session_id: str, source_docs=None):
    """Enregistre la requête dans un fichier texte."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_session_id = "".join([c for c in session_id if c.isalnum() or c in ('-', '_')]).rstrip()
    filename = os.path.join(log_dir, f"log_{timestamp}_{safe_session_id}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Time : {timestamp}\n")
        f.write(f"Session ID : {session_id}\n")
        f.write(f"Durée : {exec_time:.2f} sec\n")
        f.write(f"Model Embedding : {model_embedding}\n")
        f.write(f"Params : K={k_chunk}, Size={size}\n")
        f.write("-" * 20 + "\n")
        f.write("Prompt Utilisateur :\n")
        f.write(prompt.strip() + "\n\n")
        f.write("Réponse Assistant :\n")
        f.write(answer.strip() + "\n")

        if source_docs:
            f.write("\n=== CHUNKS UTILISÉS ===\n")
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get("source", "inconnu")
                extrait = doc.page_content.strip()
                f.write(f"\n[{i+1}] Source : {source}\n")
                f.write(f"Extrait : {extrait}...\n")

@app.post("/query")
async def ask(query: Query):
    """
    Endpoint principal compatible Unreal Engine.
    Attend un JSON : {"prompt": "...", "session_id": "..."}
    """
    result = compute_rag(query.prompt, query.session_id)
    return result


def compute_rag(prompt: str, session_id: str):
    t0 = time.time()
    print(f"session id : {session_id}")
    history_context = get_formatted_history(session_id)
    print(f"[{session_id}] Contexte historique injecté ({len(SESSIONS.get(session_id, []))} échanges)")

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
        "history_depth": len(SESSIONS[session_id])
    }


@app.post("/reset")
async def reset_history(request: ResetRequest):
    """
    Réinitialise l'historique de conversation pour une session donnée.
    """
    session_id = request.session_id
    
    if session_id in SESSIONS:
        # On supprime l'entrée du dictionnaire
        del SESSIONS[session_id]
        msg = f"Historique pour la session '{session_id}' a été effacé avec succès."
        print(f"[RESET] {msg}")
    else:
        msg = f"Aucun historique trouvé pour la session '{session_id}' (déjà vide)."
        print(f"[RESET] {msg}")

    return {
        "status": "success",
        "message": msg,
        "session_id": session_id
    }