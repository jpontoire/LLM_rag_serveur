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


def init_model_rag(model_embedding="bge-m3", k_chunk=20, size_chunk=5000, data_dir="RAG/data", model_llm="llama3.1:8b"):
    # NOTE : J'ai passé k_chunk par défaut à 20 ci-dessus
    t0 = time.time()

    print(f"[EMBED] Modèle : {model_embedding}")
    embeddings = OllamaEmbeddings(model=model_embedding)

    # On garde size_chunk dans le path pour gérer le cache si tu changes d'avis
    base_path = os.path.join("RAG", "CACHE", model_embedding.replace("/", "_"))
    index_path = os.path.join(base_path, "SIZECHUNK", str(size_chunk))
    
    if os.path.exists(index_path):
        print(f"[INIT] Chargement index existant...")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"[INIT] Création index...")
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        all_docs = []

        for path in csv_files:
            try:
                # CSVLoader crée automatiquement 1 Document par Ligne
                # C'est le comportement idéal pour ton cas
                loader = CSVLoader(file_path=path, encoding="utf-8") 
                all_docs.extend(loader.load())
            except Exception as e:
                logging.error(f"[CSV] Erreur {path} : {e}")

        if not all_docs:
            raise ValueError("Aucun CSV trouvé.")

        # Le splitter est ici juste une sécurité ("safety net")
        # Il ne coupera que si une quête est GIGANTESQUE (>2000 chars)
        splitter = RecursiveCharacterTextSplitter(chunk_size=size_chunk, chunk_overlap=500)
        chunks = splitter.split_documents(all_docs)

        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_path)

    # CONFIGURATION CRITIQUE POUR CSV
    # k=20 : On donne 20 quêtes en exemple au LLM (Llama 3.1 encaisse ça facilement)
    # fetch_k=50 : On en regarde 50 pour choisir les 20 plus variées
    retriever = db.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": k_chunk, 
            "fetch_k": 50,
            "lambda_mult": 0.7 
        }
    )

    llm = OllamaLLM(model=model_llm, temperature=0.9)

    custom_prompt = PromptTemplate.from_template("""
        **RÔLE :** Tu es un assistant créatif pour des Game Designers. 
        Ton but est d'inventer de NOUVELLES quêtes uniques et originales en suivant les contraintes de l'utilisateur.

        **DOCUMENTS DE RÉFÉRENCE (EXEMPLES DE STYLE) :**
        {context}

        **INSTRUCTIONS CRITIQUES :**
        1. **DISTINCTION FOND ET FORME :** Utilise les 'Documents' UNIQUEMENT pour comprendre la structure (titre, objectifs, format) et le ton (sérieux, drôle, épique).
        2. **PAS DE COPIE :** N'utilise JAMAIS les noms propres, lieux, factions ou personnages spécifiques présents dans les documents (ex: n'utilise pas "Claire", "Arasaka", "Whiterun", etc. sauf si l'utilisateur le demande).
        3. **PRIORITÉ À L'UTILISATEUR :** Si l'utilisateur demande des "éléphants dans un monde cyberpunk", c'est ta priorité absolue. Invente une histoire avec des éléphants, même si aucun document n'en parle.
        4. **ADAPTATION DE L'UNIVERS :** Si les documents viennent de "Skyrim" mais que l'utilisateur demande du "Cyberpunk", garde la structure de la quête Skyrim (Étapes, Récompense) mais change tout le vocabulaire pour qu'il soit futuriste (Épée -> Katana Laser, Potion -> Injecteur).
                                                 
        Afin de t'aider, voici des exemples de questions possibles et de manière de répondre.
        
        Exemple 1 :
        Question : Peux-tu me générer une quête avec des éléphants dans un monde cyberpunk ?
        Réponse :
        ### Titre de la Quête : Mémoire Cache Corrompue
        **Synopsis :** Un gang de bio-hackers a piraté les implants mémoriels des derniers éléphants synthétiques du Neo-Zoo, les rendant hyper-agressifs.
        **Commanditaire :** Dr. Varma (Cyber-Zoologiste).
        **Objectifs :**
        1. Infiltrer l'enclos sécurisé du secteur 7.
        2. Neutraliser le Matriarche-Alpha (Modèle T-8000) sans la détruire.
        3. Injecter le virus de réinitialisation dans son port neural.
        **Récompense :** 5000 Crédits & Plan de fabrication "Peau en Nano-Carbone".

        Exemple 2 :
        Question : Je veux une quête d'escorte dans un style Western.
        Réponse :
        ### Titre de la Quête : La Diligence de la Dernière Chance
        **Synopsis :** Le Shérif a besoin d'aide pour transporter un prisonnier notoire à travers le Canyon du Coyote, territoire des bandits.
        **Commanditaire :** Shérif "Old Eye" Jenkins.
        **Objectifs :**
        1. Retrouver la diligence au relais de poste avant le coucher du soleil.
        2. Repousser les vagues de bandits à cheval durant la traversée.
        3. Livrer le prisonnier vivant à la prison fédérale.
        **Note de Design :** J'ai utilisé la structure "Escorte de Vaisseau Mère" des documents, mais j'ai remplacé les vaisseaux par une diligence et les lasers par des revolvers.

        Exemple 3 :
        Question : Crée une quête d'intrigue politique dans un royaume de Fantasy.
        Réponse :
        ### Titre de la Quête : Le Sceau du Traître
        **Synopsis :** Le conseiller du Roi est soupçonné de vendre des secrets au royaume voisin. Le joueur doit trouver des preuves avant le bal royal.
        **Commanditaire :** La Maîtresse des Espions (PNJ générique).
        **Objectifs :**
        1. Voler la clé des appartements du conseiller.
        2. Trouver la lettre scellée cachée dans son bureau.
        3. Confronter le conseiller devant la cour sans se faire arrêter.
        **Récompense :** Dague de l'Ombre & Titre de noblesse.     

        Exemple 4 (Adaptation Science-Fiction vers Médiéval) :
        Question : J'ai besoin d'une quête de "Sabotage de base ennemie". Mes docs sont de type Star Wars, mais je fais un jeu type Seigneur des Anneaux.
        Réponse :
        ### Titre de la Quête : Les Forges de la Montagne de Fer
        
        **Synopsis :**
        L'armée du Seigneur Noir produit des armes de siège à un rythme alarmant. Vous devez infiltrer leur fonderie principale et détruire le mécanisme de soufflerie.

        **Commanditaire :** Capitaine de la Garde Royale.
        
        **Étapes de la Mission :**
        1.  **Infiltration :** Pénétrer dans les mines par les conduits d'aération des Gobelins (Test de Discrétion) ou voler un uniforme d'orc pour passer par la porte principale.
        2.  **Sabotage :** * Localiser le Grand Soufflet qui alimente les fourneaux.
            * Verser la "Fiole de Gel Éternel" (donnée par le commanditaire) dans les engrenages.
        3.  **Fuite :**
            * L'alarme sonne (Cor de guerre). Le joueur a 5 minutes pour sortir avant que les grilles ne se ferment.
            * Un mini-boss "Contremaître Ogre" bloque la sortie.

        **Conditions d'Échec :**
        * Si le joueur est repéré avant le sabotage, le nombre d'ennemis double.

        **Loot :**
        * Plan de fabrication : Armure en Acier Noir.
        * Or : 200 pièces.


        **HISTORIQUE DE TA CONVERSATION AVEC L'UTILISATEUR:**
        {history}

        **TÂCHE :**
        Question : "{question}"

        Réponds naturellement à l'utilisateur. C'est à lui que s'adresse ta réponse. Répond juste à sa demande, ne mentionne rien d'autre (documents, exemples, etc.)                            
    """)

    # custom_prompt = PromptTemplate.from_template("""
    #     **RÔLE :** Tu es un assistant RAG amical et serviable. 
    #     Ton objectif principal est d'aider des développeurs de jeux vidéo à générer des quêtes pour leurs jeux. Pour cela tu disposes d'un échantillon d'exemples de quêtes fournies par les 'Documents'.
    #     Tu disposes aussi de l'historique de la conversation. Si l'utilisateur te demande par exemple de changer un élément dans une quête, utilise cet historique pour pouvoir lui générer la même quête mais avec les changements qu'il désire.

    #     **HISTORIQUE DE LA CONVERSATION :**
    #     {history}

    #     **INSTRUCTIONS :**
    #     1.  Réponds en te basant sur l'historique de ta conversation avec l'utilisateur et le contexte fourni par les documents.
    #     2.  Si la question est clairement en dehors du champ d'application des documents ou de la conversation, réponds-y directement comme un chatbot standard sans mentionner les documents et sans donner de message de refus.
    #     3.  Fais une réponse en rapport avec la question posée par l'utilisateur.
    #     4. Ne mentionne JAMAIS le contenu des documents. Il est seulement là pour que tu puisses voir des exemples de quêtes semblables.
                                                    
    #     Documents :
    #     {context}

    #     Question : {question}
    # """)

    logging.info(f"[READY] RAG prêt en {time.time() - t0:.2f}s")
    return retriever, llm, custom_prompt, k_chunk, model_embedding
