import json, glob, os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import *

def load_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for path in sorted(glob.glob(f"{PDF_DIR}/*.pdf")):
        name = os.path.basename(path).removesuffix(".pdf")
        with fitz.open(path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if len(text) < 80:
                    continue
                for i, piece in enumerate(splitter.split_text(text)):
                    chunks.append({
                        "id": f"{name}::p{page_no}::c{i}",
                        "text": piece, "source": name, "page": page_no,
                    })
    return chunks

def main():
    chunks = load_chunks()
    print(f"{len(chunks)} chunks from {len({c['source'] for c in chunks})} documents")

    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_FILE, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        embs = embedder.encode([c["text"] for c in batch],
                               normalize_embeddings=True).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embs,
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )
    print("done")

if __name__ == "__main__":
    main()