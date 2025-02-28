from rag_embedding import get_completion, prompt_template
from rag_retrieval_vector import vector_db
import os

"""
Step 5: RAG Based ChatBot
"""
class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. Semantic Search
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. Build Prompt with Search Results
        prompt = f"{prompt_template}\n\nContext:\n{search_results['documents'][0]}\n\nQuery:\n{user_query}"

        # 3. Call LLM API
        response = self.llm_api(prompt)
        return response



if __name__ == "__main__":
    import sys

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        sys.exit(1)

    # Initialize the bot
    try:
        bot = RAG_Bot(vector_db, llm_api=get_completion)
        user_query = "How many parameters does Llama 2 have?"
        response = bot.chat(user_query)
        print(response)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
