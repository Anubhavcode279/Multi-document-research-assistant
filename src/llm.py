from pydantic import BaseModel, Field
from src.config import *

def make(kind: str):
    model = SMART_MODEL if kind == "smart" else FAST_MODEL
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0, num_ctx=NUM_CTX,
                          num_predict=1024, keep_alive="15m")
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=6)
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0, max_retries=6)
    raise ValueError(LLM_PROVIDER)

smart = make("smart")
fast = make("fast")

class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="1 to 3 self-contained sub-questions. Each must make sense on its own.")

class Grade(BaseModel):
    relevant: list[int] = Field(description="Numbers of the passages that help answer the question")
    sufficient: bool = Field(description="True if the relevant passages fully answer the question")
    rewritten_query: str = Field(default="", description="A better search query if not sufficient, else empty")

class Check(BaseModel):
    grounded: bool = Field(description="True only if every claim is supported by the context")
    answers_question: bool = Field(description="True if the answer addresses the whole question")
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: str = Field(default="", description="What to fix")

planner_llm = smart.with_structured_output(Plan)
grader_llm = fast.with_structured_output(Grade)
checker_llm = smart.with_structured_output(Check)

def safe_invoke(chain, prompt, fallback, tries=2):
    """Small local models sometimes return malformed JSON. Retry, then fall back."""
    for _ in range(tries):
        try:
            res = chain.invoke(prompt)
            if res is not None:
                return res
        except Exception as e:
            print("  [structured call failed]", str(e)[:120])
    return fallback

