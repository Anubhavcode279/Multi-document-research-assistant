from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:3b", temperature=0)
print(llm.invoke("What is a vector database?").content)