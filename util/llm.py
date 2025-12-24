import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def load_llm(
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.5,
):
    if os.getenv("DEEP_RESEARCH_AGENT_ENV") is None:
        load_dotenv()
    if api_key is None:
        api_key = os.getenv("LLM_API_KEY")
    if base_url is None:
        base_url = os.getenv("LLM_BASE_URL")
    if model_name is None:
        model_name = os.getenv("LLM_MODEL_NAME", "")

    llm = ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
    return llm