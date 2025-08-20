from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

documents = [
    "Delhi is the capital of India."
    "Paris is the capital of France."
    "Moscow is the capital of Russia."
]
result = embedding.embed_documents(documents)

print(str(result))


