import streamlit as st
from outils_rag import (
    get_collection,
    reformuler_question,
    generer_reponse,
    extraire_texte_pdf,
    ingerer_texte
)

# Configuration de la page
st.set_page_config(page_title="Mon Assistant IA", page_icon="🤖", layout="wide")

st.title("🤖 Mon Assistant IA Documentaire")
st.caption("Alimenté par ChromaDB et Llama 3.1 via Groq")

# Initialisation de la collection ChromaDB
collection = get_collection()

# Initialisation de l'historique de discussion
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRE LATÉRALE : UPLOAD DE DOCUMENTS ---
with st.sidebar:
    st.header("📄 Documents")
    fichiers_uploades = st.file_uploader(
        "Ajouter des documents", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if fichiers_uploades:
        for fichier in fichiers_uploades:
            # Traitement selon l'extension
            if fichier.name.endswith(".pdf"):
                contenu = extraire_texte_pdf(fichier)
            else:
                contenu = fichier.read().decode("utf-8")
            
            # Ingestion dans la base vectorielle
            nb_chunks = ingerer_texte(contenu, fichier.name, collection)
            st.success(f"✅ '{fichier.name}' ajouté ({nb_chunks} blocs)")

# --- AFFICHAGE DE L'HISTORIQUE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- GESTION DE LA QUESTION UTILISATEUR ---
if prompt := st.chat_input("Pose ta question ici..."):
    # 1. Afficher le message de l'utilisateur
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Recherche et réflexion en cours..."):
            # 2. Reformuler la question si besoin (mémoire de suivi)
            question_reformulee = reformuler_question(prompt, st.session_state.messages[:-1])
            
            # 3. Recherche de contexte dans ChromaDB
            resultats = collection.query(query_texts=[question_reformulee], n_results=3)
            
            contexte = ""
            if resultats and resultats.get("documents") and resultats["documents"][0]:
                contexte = "\n\n".join(resultats["documents"][0])
            
            # 4. Génération de la réponse via Groq
            reponse = generer_reponse(prompt, contexte)
            
            # 5. Affichage de la réponse
            st.markdown(reponse)
            
    # Enregistrement dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse})
