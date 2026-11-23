from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 1. 분류 체인 정의
template = """
주어진 사용자 질문을 `수학`, `과학`, 또는 `기타` 중 하나로 분류하세요.
이외의 답변은 허용하지 않습니다.
<question>
{question}
</question>

<answer example>
수학
</answer example>
"""

classifier_prompt = PromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-5-nano")
output_parser = StrOutputParser()

classifier_chain = classifier_prompt | model | output_parser
result = classifier_chain.invoke({"question": "2 + 2는 무엇인가요?"})

# print(result) # 수학

# 2. 주제별 체인 생성

# 수학 체인
math_prompt = PromptTemplate.from_template(
    """
    당신은 수학 전문가입니다.
    항상 다음과 같이 답변을 시작합니다. "피타고라스께서 말씀하시기를..."

    다음 질문에 답변하시요:

    질문: {question}
    답변:
    """
)
math_chain = math_prompt | model | output_parser

# 과학 체인
science_prompt = PromptTemplate.from_template(
    """
    당신은 과학 전문가입니다.
    항상 다음과 같이 답변을 시작합니다. "뉴턴께서 말씀하시기를..."

    다음 질문에 답변하시요:

    질문: {question}
    답변:
    """
)
science_chain = science_prompt | model | output_parser

# 일반 체인
general_prompt = PromptTemplate.from_template(
    """
    당신은 일반 상식 전문가입니다.
    항상 다음과 같이 답변을 시작합니다. "부모님께서 말씀하시기를..."

    다음 질문에 답변하시요:

    질문: {question}
    답변:
    """
)
general_chain = general_prompt | model | output_parser

# 3. route 함수 정의
def route(info): 
    if "수학" in info["topic"]: 
        return math_chain
    elif "과학" in info["topic"]: 
        return science_chain
    else: 
        return general_chain

# 4. 최종 체인 
full_chain =(
    {"topic": classifier_chain,
     "question": lambda x: x["question"]
     }
    | RunnableLambda(route)
    | StrOutputParser()
)

final_response = full_chain.invoke({"question": "2 + 2는 뭔가요?"})

print(final_response)
"""
피타고라스께서 말씀하시기를 2와 2를 더하면 4이니라.
"""