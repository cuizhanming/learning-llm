import calendar
import dateutil.parser as parser
from datetime import date
from langchain.tools import Tool, tool
from langchain.agents import load_tools
from langchain import hub
from dotenv import load_dotenv
load_dotenv()
tools = load_tools(["serpapi"])

# 自定义工具
@tool("weekday")
def weekday(date_str: str) -> str:
    """Convert date to weekday name"""
    d = parser.parse(date_str)
    return calendar.day_name[d.weekday()]

tools += [weekday] ## 将自定义的tool添加到tools数组中
from langchain_openai import ChatOpenAI
# 获取要使用的提示
llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    openai_api_key='sk-31bb7a65dd4047aba9b14a95c08be52c',
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

import json

# 下载一个现有的 Prompt 模板
prompt = hub.pull("hwchase17/react")

from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "周杰伦生日是那天 是星期几？"})