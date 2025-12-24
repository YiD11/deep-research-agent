import json
import os
from functools import lru_cache
from typing import List

from langchain.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import db
from db.file import FileStorageManager


@lru_cache
def get_mcp_config():
    mcp_file_path = os.getenv("MCP_FILE_PATH", "./config/mcp.json")
    with open(mcp_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def get_mcp_client():
    return MultiServerMCPClient(get_mcp_config())


def create_retrieve_file_chunks_tool(manager: FileStorageManager):
    @tool(name_or_callable="retrieve_file_chunks")
    def retrieve_file_chunks(ids: List[str]):
        """Retrieve full content chunks (with more extensive data than vector sub-chunks) by their IDs.

        Args:
            ids: List of file chunk IDs to retrieve.
        """
        results = manager.load_batch(ids)
        return results

    return retrieve_file_chunks


def create_vector_search_tool(collection_name: str, top_k: int = 5):
    @tool(name_or_callable="search_vector_chunk")
    def search_vector_chunk(query: str):
        """Search for relevant vector chunks based on a query.

        Args:
            query: The search query string.
        """
        vector_db_manager = db.get_vector_db_manager()
        vector_store = vector_db_manager.get_collection(collection_name)
        results = vector_store.similarity_search(query, k=top_k, score_threshold=0.5)
        return results

    return search_vector_chunk

async def get_all_mcp_tools() -> list[BaseTool]:
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    return mcp_tools


if __name__ == "__main__":
    import asyncio
    tools = asyncio.run(get_all_mcp_tools())
    s = json.dumps(tools, indent=2)
    print(s)
