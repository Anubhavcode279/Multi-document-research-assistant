import os, time, requests

PAPERS = {
    "RAG-Lewis": "2005.11401",
    "DPR": "2004.04906",
    "REALM": "2002.08909",
    "ColBERT": "2004.12832",
    "HyDE": "2212.10496",
    "Self-RAG": "2310.11511",
    "CRAG": "2401.15884",
    "FLARE": "2305.06983",
    "Adaptive-RAG": "2403.14403",
    "RAG-Fusion": "2402.03367",
    "Lost-in-the-Middle": "2307.03172",
    "RAG-Survey": "2312.10997",
}

os.makedirs("data/pdfs", exist_ok=True)
for name, arxiv_id in PAPERS.items():
    path = f"data/pdfs/{name}.pdf"
    if os.path.exists(path):
        continue
    r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60)
    if r.ok:
        open(path, "wb").write(r.content)
        print("downloaded", name)
    else:
        print("FAILED", name, r.status_code)
    time.sleep(3)   # be polite to arXiv
