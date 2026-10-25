import json, re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import *

def tok(text):
    return re.findall(r"\w+", text.lower())

def rrf(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection("docs")
        with open(CHUNKS_FILE) as f:
            rows = [json.loads(line) for line in f]
        self.chunks = {r["id"]: r for r in rows}
        self.ids = [r["id"] for r in rows]
        self.bm25 = BM25Okapi([tok(r["text"]) for r in rows])

    def search(self, query, fetch_k=20, final_k=FINAL_K, hybrid=True, rerank=True):
        q_emb = self.embedder.encode(QUERY_PREFIX + query,
                                     normalize_embeddings=True).tolist()
        vec_ids = self.col.query(query_embeddings=[q_emb], n_results=fetch_k)["ids"][0]

        if hybrid:
            scores = self.bm25.get_scores(tok(query))
            bm_ids = [self.ids[i] for i in np.argsort(scores)[::-1][:fetch_k]]
            candidates = rrf([vec_ids, bm_ids])[:20]
        else:
            candidates = vec_ids

        if rerank:
            pairs = [(query, self.chunks[c]["text"]) for c in candidates]
            rr = self.reranker.predict(pairs)
            order = np.argsort(rr)[::-1][:final_k]
            return [{**self.chunks[candidates[i]], "score": float(rr[i])} for i in order]
        return [{**self.chunks[c], "score": 0.0} for c in candidates[:final_k]]