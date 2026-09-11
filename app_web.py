import ollama
import streamlit as st

from outils_rag import get_collection, ingerer_texte, extraire_texte_pdf, reformuler_question

MODELE = "llama3.2"
N_RESULTATS = 3
SEUIL_PERTINENCE = 1.2  # distance ChromaDB max pour qu'un bloc soit jugé pertinent — à calibrer, voir le README

st.set_page_config(page_title="Mon IA Locale", page_icon="🤖", layout="centered")
st.title("🤖 Mon Assistant IA Local (RAG)")
st.caption("Alimenté par ChromaDB et Llama 3.2 via Ollama")

# --- CONNEXION À CHROMADB ---
try:
    collection = get_collection()
except Exception as e:
    st.error(f"Impossible de se connecter à ChromaDB : {e}")
    st.stop()

# --- SIDEBAR : AJOUT ET SUIVI DES DOCUMENTS ---
with st.sidebar:
    st.header("📚 Documents")

    if collection.count() == 0:
        st.info(
            "Ta base est vide. Ajoute un document ci-dessous, ou lance "
            "`python database.py` pour charger les 3 documents de test."
        )

    fichiers = st.file_uploader(
        "Ajouter des documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if fichiers:
        for f in fichiers:
            try:
                if f.name.endswith(".pdf"):
                    texte = extraire_texte_pdf(f)
                else:
                    texte = f.read().decode("utf-8")
                n = ingerer_texte(texte, f.name, collection)
                st.success(f"✅ {f.name} ({n} blocs)")
            except Exception as e:
                st.error(f"❌ Échec sur {f.name} : {e}")

    with st.expander(f"Voir les sources ({collection.count()} blocs en base)"):
        donnees = collection.get(include=["metadatas"])
        sources = sorted({m.get("source", "?") for m in donnees["metadatas"] if m})
        if sources:
            for s in sources:
                st.write(f"- {s}")
        else:
            st.write("Aucun document pour l'instant.")

# --- HISTORIQUE DE CONVERSATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- NOUVELLE QUESTION ---
if prompt := st.chat_input("Pose ta question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        contexte_pertinent = []
        try:
            with st.status("🔎 Recherche du contexte pertinent...") as status:
                # Historique AVANT ce message, pour reformuler une question de suivi
                requete = reformuler_question(prompt, st.session_state.messages[:-1], modele=MODELE)
                results = collection.query(query_texts=[requete], n_results=N_RESULTATS)
                documents = results["documents"][0]
                distances = results["distances"][0]
                contexte_pertinent = [d for d, dist in zip(documents, distances) if dist < SEUIL_PERTINENCE]
                status.update(label="🔎 Recherche terminée", state="complete")

            contexte = "\n\n".join(contexte_pertinent) if contexte_pertinent else "Aucun contexte pertinent trouvé."

            full_prompt = f"""
Tu es un assistant IA contextuel et poli.

Règles strictes :
1. Si l'utilisateur te salue simplement (ex: "bonjour", "coucou", "wesh", "salut"), réponds poliment et demande-lui comment tu peux l'aider avec ses documents.
2. Pour toute question d'information, appuie-toi UNIQUEMENT sur le CONTEXTE ci-dessous.
3. Si le CONTEXTE ne contient pas l'information, dis clairement : "Je n'ai pas cette information dans mes données."

CONTEXTE :
{contexte}

QUESTION DE L'UTILISATEUR :
{prompt}
"""
            stream = ollama.chat(
                model=MODELE,
                messages=[{"role": "user", "content": full_prompt}],
                stream=True,
            )
            reponse_texte = st.write_stream(chunk["message"]["content"] for chunk in stream)

            if contexte_pertinent:
                with st.expander("📄 Sources utilisées pour cette réponse"):
                    for d in contexte_pertinent:
                        st.write(d[:300] + ("…" if len(d) > 300 else ""))

        except Exception as e:
            reponse_texte = "Désolé, une erreur s'est produite. Vérifie qu'Ollama tourne bien (`ollama serve`)."
            st.error(f"{reponse_texte}\n\nDétail technique : {e}")

    st.session_state.messages.append({"role": "assistant", "content": reponse_texte})
