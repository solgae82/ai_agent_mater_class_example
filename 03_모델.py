from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("{topic}에 대해 {level} 수준으로 설명")

model = ChatOpenAI(
    model="gpt-5-nano"
    ,temperature=0.8
    ,max_completion_tokens=2048
    ,max_retries=2
    )

output_parser = StrOutputParser()

chain = prompt | model | output_parser
response = chain.invoke({"topic":"인공지능", "level":"초등학생"})
print(type(response)) # <class 'langchain_core.messages.base.TextAccessor'> 
print(response)
"""
다음은 초등학생도 이해하기 쉬운 인공지능(AI) 설명이야.
..블라블라
"""