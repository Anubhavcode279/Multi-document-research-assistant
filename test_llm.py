# test_llm.py
from src.llm import *
print(safe_invoke(planner_llm,
      "Split into sub-questions: Compare Self-RAG and CRAG ",
      Plan(sub_questions=["fallback"])))
