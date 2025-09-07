from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2:3b")

prompt = ChatPromptTemplate.from_template(
    "Write a short story about {topic}:"
)

chain = prompt | llm
result = chain.invoke({"topic": "a time-traveling scientist"})
print(result.content)