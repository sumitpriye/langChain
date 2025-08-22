from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

# Schema
class Review(TypedDict):

    # summary:str
    # product_category:str
    # sentiment:str

    summary:Annotated[str,"A brief summary of the review"]
    product_category:Annotated[str,"Category to which the product belogs for which review has been written"]
    sentiment:Annotated[Literal['pos','neg'],"Sentiment of the review, can be either positive, negative or neutral"]
    pros:Annotated[Optional[list[str]],"List down the pros"]
    cons:Annotated[Optional[list[str]],"List down the cons"]

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""
             The story started poorly and got repetitive very quickly. Characters felt flat, and halfway through I found myself skipping pages. Definitely overhyped.
             """)

print(result)

