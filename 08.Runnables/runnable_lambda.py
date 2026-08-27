from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv() 

# def word_counter(text):
#     return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke('Hi, My name is Sumit'))




llm= HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro-0813',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template = 'Write a Joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

def word_counter(text):
    return len(text.split())

joke_generator_chain = RunnableSequence(prompt1, model, parser)


parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
})


final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic':'Elephant'})

final_result = """{} \nWordCount - {}""".format(result['joke'], result['word_count'])

print(final_result)