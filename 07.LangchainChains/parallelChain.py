from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


model1 = ChatOllama(model="llama3.2:3b")
model2 = ChatOllama(model="llama3.2:3b")

prompt1 = PromptTemplate(
    template = "Generate short notes from following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = "Generate 5 short Question answer from the following text \n  {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes & quiz into a single document \n notes-->{notes} and quiz-->{quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merged_chain = prompt3 | model1 | parser

chain = parallel_chain | merged_chain 
text = """
LangChain is a framework for developing applications powered by large language models (LLMs).
LangChain simplifies every stage of the LLM application lifecycle:
Development: Build your applications using LangChain's open-source components and third-party integrations. Use LangGraph to build stateful agents with first-class streaming and human-in-the-loop support.
Productionization: Use LangSmith to inspect, monitor and evaluate your applications, so that you can continuously optimize and deploy with confidence.
Deployment: Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Platform.LangChain implements a standard interface for large language models and related technologies, such as embedding models and vector stores, and integrates with hundreds of providers. See the integrations page for more.
The LangChain framework consists of multiple open-source libraries. Read more in the Architecture page.
langchain-core: Base abstractions for chat models and other components.
Integration packages (e.g. langchain-openai, langchain-anthropic, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.
langchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
langchain-community: Third-party integrations that are community maintained.
langgraph: Orchestration framework for combining LangChain components into production-ready applications with persistence, streaming, and other key features. See LangGraph documentation.

"""
result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()