import os
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

def get_openai_client():
    """
    Initialize and return the Azure OpenAI client
    """
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return client

def get_langchain_openai_client(model_name: Optional[str] = None):
    """
    Initialize and return a LangChain Azure OpenAI client
    """
    model = model_name or os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
    
    client = AzureChatOpenAI(
        azure_deployment=model,
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.2,
    )
    return client