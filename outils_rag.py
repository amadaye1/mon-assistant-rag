"""
Fonctions partagées du projet RAG.
Utilisées par database.py et app_web.py (et réutilisables dans ajouter_doc.py / rag.py
si tu veux les garder à jour — voir la note en bas de ce fichier).

IMPORTANT : la fonction d'embedding doit être identique partout où la collection
'mes_donnees' est ouverte. Si tu la changes un jour, il faut supprimer le dossier
chroma_db/ et tout ré-ingérer : les vecteurs de deux modèles différents ne sont
pas comparables entre eux, même s'ils ont la même dimension (ChromaDB ne détecte
pas ce genre d'incohérence tout seul).
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama

CHEMIN_DB = "./chroma_db"
NOM_COLLECTION = "mes_donnees"
MODELE_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingue, adapté au français


def get_embedding_function():
    """Fonction d'embedding partagée. Téléchargée depuis Hugging Face au premier
    lancement (nécessite une connexion internet une seule fois, puis mise en cache)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODELE_EMBEDDING)


def get_collection(creer_si_absente=True):
    """Ouvre (ou crée) la collection ChromaDB avec la bonne fonction d'embedding."""
    client = chromadb.PersistentClient(path=CHEMIN_DB)
    embedding_fn = get_embedding_function()
    if creer_si_absente:
        return client.get_or_create_collection(name=NOM_COLLECTION, embedding_function=embedding_fn)
    return client.get_collection(name=NOM_COLLECTION, embedding_function=embedding_fn)


def decouper_texte(texte, taille_chunk=200, chevauchement=30):
    """Découpage par nombre de mots avec chevauchement (identique à ta version originale)."""
    mots = texte.split()
    chunks = []
    for i in range(0, len(mots), taille_chunk - chevauchement):
        chunk = " ".join(mots[i:i + taille_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingerer_texte(texte, nom_source, collection):
    """Découpe puis ingère un texte, avec métadonnées (source + index du chunk).
    Utilise upsert() plutôt que add() : réimporter le même fichier met à jour
    ses chunks au lieu de planter sur des IDs déjà existants."""
    chunks = decouper_texte(texte)
    if not chunks:
        return 0
    ids = [f"{nom_source}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": nom_source, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def extraire_texte_pdf(fichier):
    """Extrait le texte d'un PDF. Accepte un chemin (str) ou un objet fichier
    (comme ceux renvoyés par st.file_uploader)."""
    from pypdf import PdfReader
    reader = PdfReader(fichier)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def reformuler_question(question, historique, modele="llama3.2"):
    """Réécrit la question en version autonome à partir de l'historique de conversation,
    pour que la recherche sémantique dans ChromaDB reste pertinente sur les questions
    de suivi (ex: 'Et qui l'a développé ?' après une question sur un projet précis)."""
    if not historique:
        return question
    contexte_conv = "\n".join(f"{m['role']}: {m['content']}" for m in historique[-4:])
    prompt = f"""Historique de la conversation :
{contexte_conv}

Nouvelle question : {question}

Reformule cette question pour qu'elle soit compréhensible seule, sans le reste de la conversation. Réponds UNIQUEMENT avec la question reformulée, rien d'autre."""
    reponse = ollama.chat(model=modele, messages=[{"role": "user", "content": prompt}])
    return reponse["message"]["content"].strip()


# Note : si tu gardes ajouter_doc.py et rag.py en plus de app_web.py, remplace leur
# `chromadb.PersistentClient(...)` + `get_or_create_collection(...)` par
# `from outils_rag import get_collection` puis `collection = get_collection()`,
# pour être sûr d'utiliser partout le même modèle d'embedding.
