# test_retrieval.py
from src.retriever import HybridRetriever
r = HybridRetriever()
for c in r.search("How does Self-RAG decide when to retrieve?"):
    print(c["id"], round(c["score"], 2), c["text"][:90].replace("\n", " "))

