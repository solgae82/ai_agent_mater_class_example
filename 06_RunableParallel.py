from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

first_prompt = PromptTemplate.from_template("{topic}을 주제로 농담 하나 해줘.")
second_prompt = PromptTemplate.from_template("{topic}을 주제로 시 하나 작성해줘.")

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

joke_chain = first_prompt | model | parser
poem_chain = second_prompt | model | parser

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
response = map_chain.invoke({"topic":"인공지능"})

print(response)
"""
{'joke': '왜 인공지능은 항상 농담을 짧게 할까요? 데이터가 이미 다 말해주니까요.', 
'poem': '인공지능에게 바치는 시..블라블라'}
"""