from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import JsonOutputParser

# 프롬프트 템플릿
template ="""
당신은 K-POP 정보를 제공하는 AI입니다. 그룹의 멤버 이름을 말해주세요.
K-POP Group : {name}

FORMAT : 
{format}
"""


###################
# 출력파서(1) , StrOutputParser
###################

output_parser = StrOutputParser()

prompt = PromptTemplate(
    template = template,
    partial_variables={
        "format": "문자열로 출력"
    }
)

model = ChatOpenAI(model="gpt-5-nano")

chain = prompt | model | output_parser

response = chain.invoke({"name":"블랙핑크"})
print(type(response)) # <class 'langchain_core.messages.base.TextAccessor'> 
print(response)
"""
지수, 제니, 로제, 리사
"""

###################
# 출력파서(2) , CommaSeparatedListOutputParser
###################

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
"""
Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`
"""

prompt = PromptTemplate(
    template = template,
    partial_variables={
        "format": format_instructions
    }
)

model = ChatOpenAI(model="gpt-5-nano")

chain = prompt | model | output_parser

response = chain.invoke({"name":"블랙핑크"})
print(type(response)) # <class 'list'> 
print(response)
"""
['Jisoo', 'Jennie', 'Rosé', 'Lisa']
"""

###################
# 출력 파서(3), JsonOutputParser
###################
output_parser = JsonOutputParser()
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
"""
Return a JSON object.
"""

prompt = PromptTemplate(
    template = template,
    partial_variables={
        "format": format_instructions
    }
)

model = ChatOpenAI(model="gpt-5-nano")

chain = prompt | model | output_parser

response = chain.invoke({"name":"블랙핑크"})
print(type(response)) # <class 'dict'>
print(response)
"""
{'group': 'BLACKPINK', 'members': ['Jisoo', 'Jennie', 'Rosé', 'Lisa']}
"""

###################
# 출력 파서(4), JsonOutputParser
###################
output_parser = JsonOutputParser()

prompt = PromptTemplate(
    template = template,
    partial_variables={
        "format": """
Return a JSON object like this: 
{
    "answer": "질문에 대한 답변",
    "source" : "답변 생성에 활용한 출처"
}
"""
    }
)

model = ChatOpenAI(model="gpt-5-nano")

chain = prompt | model | output_parser

response = chain.invoke({"name":"블랙핑크"})
print(type(response)) # <class 'dict'>
print(response)
"""
{'answer': '지수, 제니, 로제, 리사', 'source': 'YG 엔터테인먼트 공식 프로필'}
"""