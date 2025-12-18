from fastapi import FastAPI, Query
import json, numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()
items = json.load(open("data/items.json"))
embs = np.load("data/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/recommend")
def recommend(q: str, top_k: int = 5):
    q_emb = model.encode([q], normalize_embeddings=True)[0]
    sims = np.dot(embs, q_emb)
    idxs = np.argsort(-sims)[:top_k]
    results = [{"id": items[i]["id"], "name": items[i]["name"], "score": round(float(sims[i]), 4)} for i in idxs]
    return {"query": q, "results": results}