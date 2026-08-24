from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# llm= HuggingFaceEndpoint(
#     repo_id='openai/gpt-oss-120b',
#     task='text-generation')
# model = ChatHuggingFace(llm=llm)


model  = ChatGoogleGenerativeAI(model='gemini-3.7-flash')

# Schema
class Review(BaseModel):

    summary:                str =                   Field(description = "A brief summary of the review")
    product_category:       str =                   Field(description = "Category to which the product belogs for which review has been written")
    sentiment:              Literal['pos','neg'] =  Field(description="Sentiment of the review, can be either positive, negative or neutral")
    pros:                   Optional[list[str]] =   Field(default=None, description="List down the pros")
    cons:                   Optional[list[str]] =   Field(default=None, description="List down the cons")

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""
             The story started poorly and got repetitive very quickly. Characters felt flat, and halfway through I found myself skipping pages. Definitely overhyped.
             """)

print(result.summary)

print(result)

