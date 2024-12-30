from langserve import RemoteRunnable

# joke_chain = RemoteRunnable("http://localhost:8080/joke/")
# print(joke_chain.invoke({"topic": "小明"}).content)


chat_chain = RemoteRunnable("http://localhost:8000/chat/")
print(chat_chain.invoke("Llama 2有多少参数").content)