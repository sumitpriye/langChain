from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# result = embedding.embed_query("Delhi is the capital of india.")

documents = [
    "Delhi is the capital of India."
    "Paris is the capital of France."
    "Moscow is the capital of Russia."
]
result = embedding.embed_documents(documents)

print(str(result))


