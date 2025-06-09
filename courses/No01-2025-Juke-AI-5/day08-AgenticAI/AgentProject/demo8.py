from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# 将初始化工具，让它提供答案而不是文档
tools = [TavilyAnswer(max_results=1, name="Intermediate Answer", description="Answer Search")]

# 初始化大模型
llm = ChatOpenAI(temperature=0, model="gpt-4")

# 获取使用提示 可以修改此提示
prompt = hub.pull("hwchase17/self-ask-with-search")

# 使用搜索代理构建自助询问
agent = create_self_ask_with_search_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行程序
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 运行代理
agent_executor.invoke({"input": "中国有哪些省份呢?用中文回复"})