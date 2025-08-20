from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()
llm= HuggingFacePipeline.from_model_id(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100)
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result)
print(result.content)