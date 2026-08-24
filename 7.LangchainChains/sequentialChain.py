from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1 = PromptTemplate(
    template = "Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary from the following text \n  {text}",
    input_variables=['text']
)

llm = ChatOllama(model="llama3.2:3b")

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm  | parser

result = chain.invoke({"topic": "benefits of exercise"})

print(result)

chain.get_graph().print_ascii()