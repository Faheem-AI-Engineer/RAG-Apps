import os
from dotenv import load_dotenv
import nltk
nltk.download('punkt_tab')

load_dotenv()

PINECONE_API_KEY = os.getenv('Pinecone_One_time_key')
INDEX_NAME = "hybrid-search-rag"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
