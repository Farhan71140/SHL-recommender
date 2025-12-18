import json, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
items = json.load(open("catalog/shl_catalog.json"))
texts = [f"{it['name']}. {it['description']} Skills: {', '.join(it['skills'])}." for it in items]
embs = model.encode(texts, normalize_embeddings=True)
np.save("data/embeddings.npy", embs)
json.dump(items, open("data/items.json", "w"))
print("Ingested and saved embeddings.")