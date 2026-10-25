from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.nodes import *

def fan_out(state: State):
    return [Send("research", {"sub_question": q}) for q in state["sub_questions"]]

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("research", research)
builder.add_node("synthesize", synthesize)
builder.add_node("self_check", self_check)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", fan_out, ["research"])
builder.add_edge("research", "synthesize")
builder.add_edge("synthesize", "self_check")
builder.add_conditional_edges("self_check", route_after_check,
                              {"revise": "synthesize", "end": END})

graph = builder.compile()
RUN_CONFIG = {"max_concurrency": 2}   # a local Ollama server handles ~1-2 requests at a time

def ask(question: str) -> dict:
    out = graph.invoke({"question": question, "findings": [], "revisions": 0},
                       config=RUN_CONFIG)
    out["final_answer"] = render_citations(out["answer"], out["findings"])
    return out

if __name__ == "__main__":
    import sys, time
    q = " ".join(sys.argv[1:]) or "Compare how Self-RAG and CRAG handle low-quality retrieval."
    t = time.time()
    for step in graph.stream({"question": q, "findings": [], "revisions": 0},
                             config=RUN_CONFIG, stream_mode="updates"):
        for node, update in step.items():
            if node == "planner":
                print("PLAN:", update["sub_questions"])
            elif node == "research":
                f = update["findings"][0]
                print(f"RESEARCH: {f['sub_question']!r} -> {len(f['chunks'])} chunks, sufficient={f['sufficient']}")
            elif node == "self_check":
                print("CHECK:", update["check"])
    out = ask(q)
    print("\nANSWER:\n", out["final_answer"], f"\n\n({time.time()-t:.0f}s)")