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

"""
Fonctions partagées du projet RAG pour le déploiement sur Streamlit Cloud (Groq API).
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

CHEMIN_DB = "./chroma_db"
NOM_COLLECTION = "mes_donnees"
MODELE_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"


def get_client_groq():
    """Initialise le client Groq à partir de la variable d'environnement GROQ_API_KEY."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("La clé GROQ_API_KEY n'est pas configurée dans les secrets.")
    return Groq(api_key=api_key)


def get_embedding_function():
    """Fonction d'embedding partagée."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODELE_EMBEDDING)


def get_collection(creer_si_absente=True):
    """Ouvre (ou crée) la collection ChromaDB."""
    client = chromadb.PersistentClient(path=CHEMIN_DB)
    embedding_fn = get_embedding_function()
    if creer_si_absente:
        return client.get_or_create_collection(name=NOM_COLLECTION, embedding_function=embedding_fn)
    return client.get_collection(name=NOM_COLLECTION, embedding_function=embedding_fn)


def decouper_texte(texte, taille_chunk=200, chevauchement=30):
    """Découpage par nombre de mots avec chevauchement."""
    mots = texte.split()
    chunks = []
    for i in range(0, len(mots), taille_chunk - chevauchement):
        chunk = " ".join(mots[i:i + taille_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingerer_texte(texte, nom_source, collection):
    """Découpe puis ingère un texte dans ChromaDB."""
    chunks = decouper_texte(texte)
    if not chunks:
        return 0
    ids = [f"{nom_source}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": nom_source, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def extraire_texte_pdf(fichier):
    """Extrait le texte d'un fichier PDF."""
    from pypdf import PdfReader
    reader = PdfReader(fichier)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def reformuler_question(question, historique):
    """Réécrit la question en version autonome avec Groq."""
    if not historique:
        return question

    client = get_client_groq()
    contexte_conv = "\n".join(f"{m['role']}: {m['content']}" for m in historique[-4:])
    prompt = f"""Historique de la conversation :
{contexte_conv}

Nouvelle question : {question}

Reformule cette question pour qu'elle soit compréhensible seule, sans le reste de la conversation. Réponds UNIQUEMENT avec la question reformulée, rien d'autre."""

    response = client.chat.completions.create(
        model="llama3-8b-8192",  # ← Vérifie bien ce nom
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generer_reponse(prompt, contexte):
    """Génère la réponse finale via l'API Groq sans répéter les salutations inutiles."""
    client = get_client_groq()
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ← Vérifie bien ce nom
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant IA spécialisé dans l'analyse de documents. "
                    "Si l'utilisateur salue uniquement (ex: bonjour), répond brièvement. "
                    "Si l'utilisateur pose une question, ne dis PAS bonjour et réponds directement. "
                    "Appuie-toi UNIQUEMENT sur le CONTEXTE fourni. "
                    "Si l'information n'est pas dans le contexte, dis clairement : 'Je n'ai pas cette information dans mes données.'"
                )
            },
            {
                "role": "user",
                "content": f"CONTEXTE:\n{contexte}\n\nQUESTION:\n{prompt}"
            }
        ]
    )
    return completion.choices[0].message.content


# Note : si tu gardes ajouter_doc.py et rag.py en plus de app_web.py, remplace leur
# `chromadb.PersistentClient(...)` + `get_or_create_collection(...)` par
# `from outils_rag import get_collection` puis `collection = get_collection()`,
# pour être sûr d'utiliser partout le même modèle d'embedding.
