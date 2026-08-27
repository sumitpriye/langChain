from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents=[
        "Virat Kohli is the modern-day run machine, feared for his aggressive batting and chase-mastering skills.",
        "MS Dhoni is celebrated as “Captain Cool,” a leader who guided India with calmness and iconic finishing ability.",
        "Rohit Sharma is known as the “Hitman,” admired for his elegant stroke play and record double centuries in ODIs.",
        "Suresh Raina is remembered as Mr. IPL, a dynamic left-hander and one of India’s finest fielders.",
        "Yashasvi Jaiswal is the rising star of Indian cricket, famed for his fearless batting and hunger for big runs."
        ]

query = "tell me about suresh Raina"

documents_embedding = embedding.embed_documents(documents)
query_embedding  = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],documents_embedding)[0]
print(scores)

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score is : ",float(score))
