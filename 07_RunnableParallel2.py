from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

# 1차 질문 답변 생성
question = "인셉션 감독이 누구인가요?"

template1 = """
당신은 영화를 추천해주는 AI 챗봇입니다. User의 질문에 대해 답변하시오.
question : {question}
"""

prompt = PromptTemplate(template=template1)
model = ChatOpenAI(model="gpt-5-nano")

chain1 = prompt | model | StrOutputParser()
answer = chain1.invoke({"question": question})

# 2차 새 질문 2개 체인 생성(recommend, argumented)
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

# 1차에서 받은 질문 다음으로 할만한 추천 질문 3개 다시 만들기
recommend_template ="""
당신은 영화를 추천해 주는 AI 챗봇입니다. 
User의 질문과 AI의 답변 다음으로 User가 질문할 만한 추천 prompt 3개를 말해주세요
User_message : {question}
AI_message : {answer}

FORMAT : 
{format}
"""

recommend_prompt = PromptTemplate(
    template=recommend_template,
    partial_variables={
        "format": format_instructions
    }
)

# 사용자 질문 증강
augmented_template ="""
당신은 사용자의 질문을 새롭게 생성하는 인공지능 비서입니다. 
User의 질문과 유사하지만 사용한 단어는 다른 질문 3개를 말해주세요.
User_message : {question}

FORMAT : 
{format}
"""

augmented_prompt = PromptTemplate(
    template=augmented_template,
    partial_variables={
        "format": format_instructions
    }
)

# model = ChatOpenAI() #쓰잘데기 없는 에러 코드, 찾음

recommend_chain = recommend_prompt | model | output_parser
augmented_chain = augmented_prompt | model | output_parser

# 새 생성 체인 2개 실행

map_chain = RunnableParallel(
    recommend_response=recommend_chain,
    augmented_question=augmented_chain
)

result = map_chain.invoke({"question": question, "answer":answer})

print(result)
"""
{'recommend_response': ['인셉션과 분위기가 비슷한 영화 5편 추천해줘', 
'크리스토퍼 놀란의 다른 작품을 장르별로 정리해 추천해줘', 
'나의 취향을 말하면 그에 맞춘 영화 추천 리스트와 간단한 소개를 제공해줘'], 
'augmented_question': ['인셉션의 감독은 누구인가요?', 
'인셉션을 연출한 사람은 누구인가요?', 
'인셉션의 연출자는 누구입니까?']}
"""