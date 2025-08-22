from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional,Literal
from pydantic import BaseModel,Field

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)


result = structured_model.invoke("""
             The story started poorly and got repetitive very quickly. Characters felt flat, and halfway through I found myself skipping pages. Definitely overhyped.
             """)

print(result)

