from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class LLMHandler:
    def __init__(self, model="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024):
        #api_key = os.getenv("GROQ_API_KEY")
        api_key = "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"
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)

    def generate_answer(self, context, query):
        prompt = f"Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion:{query}\n\nAnswer:"
        return self.llm.invoke([prompt]).content
