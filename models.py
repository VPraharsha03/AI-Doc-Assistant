from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os as OS

class Models:
    def __init__(self):
        oai_key = OS.getenv("OPENAI_API_KEY")
        hf_key = OS.getenv("HF_API_KEY")
        oai_base = "https://openrouter.ai/api/v1"
        '''
        self.embeddings_openai = OpenAIEmbeddings(
            openai_api_key=oai_key, 
            openai_api_base=oai_base,
            model_name="google/gemma-3-1b-it:free" # HuggingFace embedding models
            )
        '''
        self.embeddings_hf = HuggingFaceInstructEmbeddings(
            model_name="intfloat/e5-small-v2"
            )

        self.chat_model = ChatOpenAI(
            openai_api_key=oai_key, 
            openai_api_base=oai_base, 
            model_name="deepseek/deepseek-r1:free" # OpenRouter only supports chat models
            )