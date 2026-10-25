import operator, re
from typing import TypedDict, Annotated
from src.llm import *
from src.retriever import HybridRetriever
from src.config import *

retriever = HybridRetriever()

class State(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]   # parallel branches append here
    answer: str
    check: dict
    revisions: int

def numbered(chunks):
    return "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))

def build_context(findings):
    """Give each unique chunk a short tag (S1, S2, ...) so small models cite reliably."""
    tag_of, text = {}, ""
    for f in findings:
        text += f"\n### Sub-question: {f['sub_question']}\n"
        for c in f["chunks"]:
            if c["id"] not in tag_of:
                tag_of[c["id"]] = f"S{len(tag_of) + 1}"
            text += f"[{tag_of[c['id']]}] ({c['source']}, p.{c['page']})\n{c['text']}\n\n"
    return text, tag_of

# ---------- 1. Planner ----------
def planner(state: State):
    q = state["question"]
    plan = safe_invoke(
        planner_llm,
        "Break this research question into focused sub-questions (maximum 3). "
        "If it is already simple, return just ONE sub-question equal to the question.\n\n"
        f"Question: {q}",
        Plan(sub_questions=[q]),
    )
    subs = [s.strip() for s in plan.sub_questions if s.strip()][:MAX_SUBQUESTIONS] or [q]
    return {"sub_questions": subs}

# ---------- 2. Research (one run per sub-question) ----------
def research(payload: dict):
    sub_q = payload["sub_question"]
    query, kept, sufficient = sub_q, {}, False

    for hop in range(MAX_HOPS + 1):
        results = retriever.search(query)
        fallback = Grade(relevant=list(range(1, len(results) + 1)), sufficient=True)
        g = safe_invoke(
            grader_llm,
            f"Question: {sub_q}\n\nPassages:\n{numbered(results)}\n\n"
            "Return the numbers of passages that help answer the question. "
            "Say whether they are SUFFICIENT to answer it. If not sufficient, "
            "write a better search query using different keywords.",
            fallback,
        )
        for n in g.relevant:
            if 1 <= n <= len(results):
                kept[results[n - 1]["id"]] = results[n - 1]
        if g.sufficient:
            sufficient = True
            break
        if not g.rewritten_query or g.rewritten_query.strip() == query:
            break
        query = g.rewritten_query.strip()

    chunks = list(kept.values())[:FINAL_K + 2]
    return {"findings": [{"sub_question": sub_q, "chunks": chunks, "sufficient": sufficient}]}

# ---------- 3. Synthesize ----------
def synthesize(state: State):
    context, _ = build_context(state["findings"])
    if not any(f["chunks"] for f in state["findings"]):
        return {"answer": "I could not find relevant information in the documents."}

    feedback = ""
    chk = state.get("check")
    if chk and not (chk["grounded"] and chk["answers_question"]):
        feedback = ("\n\nA fact-checker rejected your previous answer.\n"
                    f"Problems: {chk['feedback']} {chk['unsupported_claims']}\n"
                    "Rewrite it. Remove any claim you cannot support.")

    msg = smart.invoke(
        "You are a research assistant. Answer using ONLY the context below.\n"
        "Rules:\n"
        "- After each factual claim, cite the passage tag in square brackets, like [S1] or [S2].\n"
        "- If the context only partly answers the question, say what is missing.\n"
        "- If comparing documents, state clearly which document says what.\n"
        "- Be concise.\n\n"
        f"Question: {state['question']}\n\nContext:{context}{feedback}"
    )
    return {"answer": msg.content}

# ---------- 4. Self-check ----------
def self_check(state: State):
    context, _ = build_context(state["findings"])
    fallback = Check(grounded=True, answers_question=True)   # lenient if the checker fails
    verdict = safe_invoke(
        checker_llm,
        "You are a strict fact-checker. Judge the answer ONLY against the context.\n"
        "grounded=false if any claim is not supported by the context or a citation tag "
        "points to a passage that does not support it.\n\n"
        f"Question: {state['question']}\n\nAnswer:\n{state['answer']}\n\nContext:{context}",
        fallback,
    )
    return {"check": verdict.model_dump(),
            "revisions": state.get("revisions", 0) + 1}

def route_after_check(state: State):
    ok = state["check"]["grounded"] and state["check"]["answers_question"]
    return "end" if ok or state["revisions"] > MAX_REVISIONS else "revise"

def render_citations(answer: str, findings: list[dict]) -> str:
    """Replace [S1] tags with [Paper p.N]."""
    _, tag_of = build_context(findings)
    info = {}
    for f in findings:
        for c in f["chunks"]:
            info[tag_of[c["id"]]] = f"{c['source']} p.{c['page']}"
    def repl(m):
        tags = [t.strip() for t in m.group(1).split(",")]
        return "[" + "; ".join(info.get(t, t) for t in tags) + "]"
    return re.sub(r"\[(S\d+(?:\s*,\s*S\d+)*)\]", repl, answer)