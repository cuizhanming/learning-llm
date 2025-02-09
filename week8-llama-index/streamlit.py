# pip install llama-index, llama-index-embeddings-huggingface, llama-index-llms-huggingface
# pip install streamlit

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

st.set_page_config(page_title="LlamaIndex Demo", page_icon=":llama:", layout="wide")
st.title("LlamaIndex Demo")

# Load the LLM model
@st.cache_resource
def init_model():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Settings.embed_model = embed_model

    llm = HuggingFaceLLM(
        model_name="Qwen/Qwen1.5-1.8B-Chat",
        tokenizer_name="Qwen/Qwen1.5-1.8B-Chat",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
        device_map="auto"
    )
    Settings.llm = llm

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    return query_engine

if "query_engine" not in st.session_state:
    st.session_state.query_engine = init_model()

def greet(question):
    response = st.session_state.query_engine.query(question)
    return response

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am RoboCui, What can I help you with today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am RoboCui, What can I help you with today?"}]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

def generate_llama_index_response(question):
    return greet(question)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama_index_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

# pip install streamlit
# streamlit run streamlit.py