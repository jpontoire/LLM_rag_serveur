# === init_rag_ollama.py ===
import os
import glob
import time
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# === Logging ===
log_dir = os.path.join("LOGS", "SERVER")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "rag.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def try_load_text_safe(path):
    try:
        if os.path.getsize(path) < 10:
            raise ValueError("Fichier trop petit")
        return TextLoader(path, encoding="utf-8").load()
    except UnicodeDecodeError:
        try:
            return TextLoader(path, encoding="ISO-8859-1").load()
        except Exception as e:
            logging.error(f"[TXT][ISO FAIL] {path} — {e}")
    except Exception as e:
        logging.error(f"[TXT][UTF-8 FAIL] {path} — {e}")
    return []

def init_model_rag(model_embedding="bge-m3", k_chunk=4, size_chunk=2000, data_dir="RAG/data", model_llm="llama3.1:8b"):
    t0 = time.time()

    print(f"[EMBED] Utilisation du modèle Ollama pour les embeddings : {model_embedding}")
    embeddings = OllamaEmbeddings(model=model_embedding)

    base_path = os.path.join("RAG", "CACHE", model_embedding.replace("/", "_"))
    index_path = os.path.join(base_path, "SIZECHUNK", str(size_chunk))
    os.makedirs(base_path, exist_ok=True)

    all_docs = []
    error_count = 0

    if os.path.exists(index_path):
        print(f"[INIT] Index FAISS trouvé pour {index_path}/, chargement...")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"[INIT] Aucun index FAISS pour {index_path}, création en cours...")
        txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
        pdf_files = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

        for path in txt_files:
            docs = try_load_text_safe(path)
            if docs:
                all_docs.extend(docs)
            else:
                error_count += 1
                logging.warning(f"[TXT] Échec chargement : {path}")

        for path in pdf_files:
            try:
                all_docs.extend(PyMuPDFLoader(path).load())
            except Exception as e:
                error_count += 1
                logging.error(f"[PDF] {path} — {e}")

        for path in csv_files:
            try:
                all_docs.extend(CSVLoader(path).load())
            except Exception as e:
                error_count += 1
                logging.error(f"[CSV] {path} — {e}")

        if not all_docs:
            logging.critical(f"[ERREUR] Aucun document trouvé dans {data_dir}")
            raise ValueError("Aucun document exploitable")

        print(f"[LOAD] {len(all_docs)} documents chargés, {error_count} erreurs")
        splitter = RecursiveCharacterTextSplitter(chunk_size=size_chunk, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)

        if not chunks:
            logging.critical("[SPLIT] Aucun chunk généré.")
            raise ValueError("Échec split documents")

        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_path)
        logging.info(f"[EMBED] Index FAISS créé et sauvegardé ({len(chunks)} chunks)")

    retriever = db.as_retriever(search_kwargs={"k": k_chunk})
    llm = OllamaLLM(model=model_llm)

    custom_prompt = PromptTemplate.from_template("""
        Documents :
        {context}

        Question : {question}
        Réponds de manière précise et concise.
    """)

    logging.info(f"[READY] RAG prêt en {time.time() - t0:.2f}s")
    return retriever, llm, custom_prompt, k_chunk, model_embedding
