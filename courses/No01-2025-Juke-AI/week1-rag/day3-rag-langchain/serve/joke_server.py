#!/usr/bin/env python
from IPython.terminal.shortcuts.filters import pass_through
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

# model = ChatOpenAI()
# prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
# chat_chain = prompt | model
# add_routes(
#     app,
#     chat_chain,
#     path="/joke",
# )

# Load a PDF file
loader = PyMuPDFLoader("../llama2.pdf")
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)
texts = text_splitter.create_documents([page.page_content for page in pages[:4]])

# Indexing the PDF file to FAISS
from langchain_community.vectorstores import FAISS
openAiEmbeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.from_documents(texts, openAiEmbeddings)

# Answer question with retriever
retriever = db.as_retriever(search_kwargs={"k": 1})
prompt_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model =  ChatOpenAI()
lang_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser
)

add_routes(
    app,
    lang_chain,
    path="/chat"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)