import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

#프롬프트
prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해 주세요.")

#모델
model = ChatOpenAI(model="gpt-5-nano")

#출력파서
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# invoke 메소드 출력
# response = chain.invoke({"topic":"양자역학"})
# print(response)

#stream 메소드 사용 출력, chunk 단위로 받아서 출력
#for chunk in chain.stream({"topic": "양자역학"}): 
#    print(chunk, end='')

# 다수 변수 출력
prompt2 = PromptTemplate.from_template("{topic}에 대해 {level} 수준으로 설명해 주세요.")
chain2 = prompt2 | model | output_parser

#response = chain2.invoke({"topic":"인공지능", "level":"초등학생"})
#print(response)

# 체인 결합 
chain2_dic = {"chain2_result": chain2}

prompt3 = PromptTemplate.from_template("이 정보가 유익한가요? 정보: {chain2_result}")

combine_chain = chain2_dic | prompt3 | model | output_parser
final_result = combine_chain.invoke({"topic":"인공지능","level":"초등학생"})

print(final_result)
