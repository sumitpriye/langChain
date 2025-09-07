from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Create local pipeline with a lightweight model
model_id = "llama3.2:3b"  # Small, fast model
pipe = pipeline(
    "text-generation",
    model=model_id,
    max_length=150,
    temperature=0.7,
    device=device,
    pad_token_id=50256  # Avoid warnings
)

# Create LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Write a short description about {topic}. Keep it under 100 words."
)

# Create chain
chain = prompt | llm

# Test the chain
try:
    print("Testing chain...")
    result = chain.invoke({'topic': 'Pokemon'})
    print("Result:", result)
except Exception as e:
    print(f"Error: {e}")
    
# Alternative: Simple string prompt for better compatibility
from langchain_core.prompts import PromptTemplate

simple_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Topic: {topic}\nDescription:"
)

simple_chain = simple_prompt | llm

try:
    print("\nTesting simple chain...")
    result = simple_chain.invoke({'topic': 'Pokemon'})
    print("Result:", result)
except Exception as e:
    print(f"Error: {e}")