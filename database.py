from outils_rag import get_collection, NOM_COLLECTION

collection = get_collection()

documents = [
    "Projet Alpha : Il s'agit d'un jeu vidéo de survie développé sur Roblox Studio.",
    "Projet Beta : C'est une application web codée en Python et HTML/CSS pour gérer des tâches.",
    "Règle du serveur : La clé API principale doit toujours être stockée dans un fichier .env et jamais publiée.",
]
ids = ["doc1", "doc2", "doc3"]
metadatas = [{"source": "documents_test"} for _ in documents]

collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

print(f"✅ {len(documents)} documents de test ajoutés/mis à jour dans la collection '{NOM_COLLECTION}'.")
print(f"   Nombre total de blocs en base : {collection.count()}")
