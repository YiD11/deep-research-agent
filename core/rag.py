import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List

import gradio
from langchain.agents import AgentState
from langchain.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from db import FileStorageManager, get_vector_db_manager
from pkg.graph import create_agent_graph
from util.doc import chunk_markdown
from util.llm import load_llm
from util.tool import (
    create_retrieve_file_chunks_tool,
    create_vector_search_tool,
    get_all_mcp_tools,
)

logger = logging.getLogger(__name__)


class RagSystem:
    def __init__(
        self,
        markdown_dir: str = None,
        collection_name: str = None,
        thread_id: str = None,
    ) -> None:
        logger.info("initializing RAG System")
        self.vector_store = get_vector_db_manager()
        self.collection_name = collection_name
        if self.collection_name is None:
            self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        file_store_path = os.getenv("FILE_STORE_PATH")
        self.file_store = FileStorageManager(file_store_path)

        model_name = os.getenv("LLM_MODEL_NAME", "")
        base_url = os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        llm = load_llm(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )
        mcp_tools = asyncio.get_event_loop().run_until_complete(get_all_mcp_tools())
        retrieve_file_chunks_tool = create_retrieve_file_chunks_tool(self.file_store)
        vector_search_tool = create_vector_search_tool(self.collection_name)
        all_tools = [retrieve_file_chunks_tool, vector_search_tool, mcp_tools]

        if markdown_dir is None:
            markdown_dir = os.getenv("MARKDOWN_FILE_PATH")
        self.markdown_dir = Path(markdown_dir)
        os.makedirs(self.markdown_dir, exist_ok=True)

        self.thread_id = uuid.uuid4().hex if thread_id is None else thread_id
        self.agent_checkpointer = InMemorySaver()
        self.agent_graph = create_agent_graph(
            llm=llm, tools=all_tools, checkpointer=self.agent_checkpointer
        )

    def add_documents(
        self,
        document_paths: List[str],
        progess: gradio.Progress,
    ):
        total = len(document_paths)
        document_paths = list(filter(lambda x: x.endswith(".md"), document_paths))
        pbar = progess.tqdm(iterable=range(total), total=total, desc="Adding documents")
        if not all([Path(x).exists() for x in document_paths]):
            raise FileNotFoundError("Some documents do not exist")

        success, failed = 0, 0
        for p in document_paths:
            src_path = Path(p)
            chunks, sub_chunks = chunk_markdown(src_path)
            try:
                self.file_store.save_batch(chunks)
                self.vector_store.get_collection(self.collection_name).add_documents(
                    sub_chunks
                )
                shutil.copy(src_path, self.markdown_dir / src_path.name)
                success += 1
            except Exception as e:
                failed += 1
                logger.error(f"Failed to add document {src_path}: {e}")

            pbar.update(1)

        return success, failed

    def clean_all_documents(self):
        self.file_store.clean_store()
        if self.markdown_dir.exists():
            shutil.rmtree(self.markdown_dir)
            os.makedirs(self.markdown_dir, exist_ok=True)
        self.vector_store.delete_collection(self.collection_name)

    def get_markdown_files(self):
        return list(self.markdown_dir.glob("*.md"))

    async def chat(
        self,
        message: str,
        history: List[AnyMessage],
    ):
        config = {"thread_id": self.thread_id}
        response = await self.agent_graph.ainvoke(
            AgentState(messages=[HumanMessage(content=message)]),
            config=config,
        )
        ret = str(response["messages"][-1].content)
        return ret

    async def clean_session(self):
        await self.agent_checkpointer.adelete_thread(self.thread_id)
        self.thread_id = uuid.uuid4().hex
