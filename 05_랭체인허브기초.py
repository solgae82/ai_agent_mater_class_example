import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client


# 랭스미스 API 키 가져오기
LANGSMITH_API_KEY = os.environ.get('LANGSMITH_API_KEY')

client = Client(api_key=LANGSMITH_API_KEY)
# 랭허브에서 프롬프트 가져오기, 정책이 바뀌어서 dangerously_pull_public_prompt=True 해줘야함
prompt = client.pull_prompt('hardkothari/prompt-maker', include_model=True, dangerously_pull_public_prompt=True)

# 랭허브에서 가져온 포맷에 맞게 {task},{lazy_prompt} 프롬프트 작성
task = """
반드시 한글로 작성되어야 합니다.
사용자의 질문을 읽고, 핵심 키워드를 파악해 전문 지식이 있는 사람의 질문으로 변경해주세요.
더 체계적이고, 단계적인 질문이 될 수 있도록 변경하세요.
사용자의 질문에서 벗어나서는 안됩니다.
"""

# 사용자의 질문
lazy_prompt = "강아지 눈이 이상해요"

chain = prompt | ChatOpenAI(model="gpt-5-nano") | StrOutputParser()

improved_prompt = chain.invoke({"task":task, "lazy_prompt": lazy_prompt})

print(improved_prompt) # 향상된 질문 형식으로 교체된 prompt가 출력됨
