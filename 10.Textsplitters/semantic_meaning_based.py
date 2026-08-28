from langchain_text_splitters import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = """
Artificial intelligence is transforming the way businesses operate by automating repetitive tasks and helping employees make faster decisions. At the same time, regular physical exercise plays an important role in maintaining good health, improving energy levels, and reducing stress. Many organizations are also adopting cloud computing because it allows applications and data to be accessed efficiently without maintaining extensive physical infrastructure.

Learning a new programming language can improve problem-solving skills and open up new career opportunities. Python, for example, is widely used in web development, data analysis, machine learning, and automation.
"""
# vector = embeddings.embed_query(text)


splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

result = splitter.split_text(text)
print(len(result))
print(result[0])
print("*********************************")
print(result[1])
print("*********************************")
print(result[2])