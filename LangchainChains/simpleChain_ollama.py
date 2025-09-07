from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

llm = ChatOllama(model="llama3.2:3b")

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "soul"})

print(result)

chain.get_graph().print_ascii()