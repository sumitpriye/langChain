from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


model_name = "facebook/opt-1.3b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text-generation pipeline
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_length=100,
    temperature=0.7,
    do_sample=True,
)

# Wrap it in LangChain
llm = HuggingFacePipeline(pipeline=gen_pipeline)


# Create prompt template
# prompt  = "What is the function of heart?"


parser = StrOutputParser()

# Create chain
# chain = prompt | llm | parser

result = llm.invoke("""Write a short story about a robot who learns how to paint beautiful pictures. 
Make it fun and creative.""")

print(result)


# chain.get_graph().print_ascii()