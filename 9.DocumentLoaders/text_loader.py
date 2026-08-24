from langchain_community.document_loaders import TextLoader 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro-0813',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary for the following poem {poem}',
    input_variables=['poem']
)

loader = TextLoader('9.DocumentLoaders\cosmicCanvas.txt', encoding = 'utf-8')
docs  = loader.load()

# print(docs)

# print(type(docs)) --list

# print(len(docs))

# print(docs[0])  

# print(type(docs[0]))  --Document

# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt  | model | parser
print(chain.invoke({'poem':docs[0].page_content}))