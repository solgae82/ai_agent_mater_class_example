from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

# 1. from_template 사용
prompt = PromptTemplate.from_template("{topic}에 대해 {level} 수준으로 설명")
#print(prompt)

# 2. PromptTemplate 사용
prompt2 = PromptTemplate(
    template="{topic}에 대해 {level} 수준으로 설명해 주세요",
    input_variables=["topic","level"],
    partial_variables={
        "topic":"양자역학",
        #"level":"초등학생"
    }
)

# 빠진 변수들을 체크할 수 있다
#print(prompt2)
"""
input_variables=['level'] 
input_types={} 
partial_variables={'topic': '양자역학'} 
template='{topic}에 대해 {level} 수준으로 설명해 주세요'
"""

# PromptTemplate > partial_variables 분석

prompt3 = PromptTemplate(
    template="{year} + {n} + {m}는 무엇인가요?",
    partial_variables={
        "year" : "2025" # 미리 셋팅
    }
)
#모델
model = ChatOpenAI(model="gpt-5-nano")

#출력파서
output_parser = StrOutputParser()

chain = prompt3 | model | output_parser
response = chain.invoke({"n":"3", "m":"5"})
#print(response)
"""
2033입니다.  
계산: 2025 + 3 = 2028, 2028 + 5 = 2033.
"""

# 함수 넣기 , n 값 덮어쓰기
def get_current_year():
    return str(datetime.now().year)

prompt4 = PromptTemplate(
    template="{year} + {n} + {m}는 무엇인가요?",
    partial_variables={
        "year" : get_current_year,
        "n" : "10"
    }
)
chain = prompt4 | model | output_parser
response = chain.invoke({"n":"3", "m":"5"}) # n=3 으로 덮어쓰기
print(response)
"""
$ py 02_프롬프트.py
2034입니다. 계산: 2026 + 3 = 2029, 2029 + 5 = 2034.
"""

# 3. ChatPromptTemplate, 채팅 맥락 참고해서 답변 받기
prompt5 = ChatPromptTemplate([
    ("system","당신은 강아지 안구 건강 관련 조언 AI 챗봇입니다"),
    ("human","안녕하세요! 저희 집 강아지가 눈을 자주 긁어요"),
    ("ai","그렇군요, 사용자님 강아지는 몇 살인가요?"),
    ("human","{user_input}")
])

chain = prompt5 | model | output_parser
response = chain.invoke({"user_input": "3살이에요"})
print(response)
"""
좋은 질문이에요. 3살 강아지가...블라블라
"""