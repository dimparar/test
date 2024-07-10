from langchain_community.llms import Ollama
import os

# Simulating user input
filename = input("Enter the filename to delete: ")

# Vulnerable command execution
os.system(f"rm {filename}")

llm = Ollama(model="gemma:2b")
result = llm.invoke("Tell me why the sky is blue")
print(result)

print("hello world")