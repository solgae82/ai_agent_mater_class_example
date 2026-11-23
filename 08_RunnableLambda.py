from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import CommaSeparatedListOutputParser

def length_function(word): 
    return len(word)

# RunnableLambda 체인 테스트
prompt = PromptTemplate.from_template(
    "{a} + {b}는 무엇인가요?"
)
model = ChatOpenAI(model="gpt-5-nano")
output_parser = StrOutputParser()

from langchain_core.runnables import RunnableLambda

chain = (
    {
        "a": RunnableLambda(lambda x: length_function(x["word1"])),
        "b": RunnableLambda(lambda x: length_function(x["word2"]))
    }
    | prompt | model | output_parser
)

base_response = chain.invoke({"word1": "안녕하세요", "word2": "반가워요"})
print(base_response)
"""
5 + 4는 9입니다.
"""