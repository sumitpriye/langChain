from langchain_community.document_loaders import WebBaseLoader 
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

url = 'https://en.wikipedia.org/wiki/LangChain'

loader = WebBaseLoader(url)
docs  = loader.load()

# # print(docs)

# print(type(docs)) 
# # --list

# print(len(docs))

# # print(docs[0])  

# print(type(docs[0]))  
# # --Document

# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt  | model | parser
print(chain.invoke({'question':'As of April 2023 , how many document types can langchain read ? ', 'text':docs}))