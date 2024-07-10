from langchain_community.llms import Ollama

llm = Ollama(model="gemma:2b")
result = llm.invoke("Tell me why the sky is blue")
print(result)

print("hello world waz up")
