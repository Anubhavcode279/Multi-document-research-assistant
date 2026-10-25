from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import ask

app = FastAPI()

class Q(BaseModel):
    question: str

@app.post("/ask")
def ask_endpoint(q: Q):
    out = ask(q.question)
    sources = sorted({f"{c['source']} p.{c['page']}" for f in out["findings"] for c in f["chunks"]})
    return {
        "answer": out["final_answer"],
        "sub_questions": out["sub_questions"],
        "sources": sources,
        "check": out["check"],
        "revisions": out["revisions"],
    }