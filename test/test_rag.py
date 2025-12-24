import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, List

import pytest
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault(
    "QDRANT_DENSE_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
os.environ.setdefault(
    "QDRANT_SPARSE_EMBEDDING_MODEL_NAME", "prithivida/Splade_PP_en_v1"
)
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")

from test.dummy import TEST_MARKDOWN_CONTENT

from const.document import META_SOURCE_FILE_CHUNK_ID_KEY, META_SOURCE_FILE_KEY
from db.file import FileStorageManager
from db.vector import VectorDBManager, get_vector_db_manager
from pkg.graph import create_agent_graph
from util.tool import retrieve_file_chunks, search_vector_chunk
from util.doc import chunk_markdown
from util.llm import load_llm

TEST_COLLECTION_NAME = "test_rag_collection"


@pytest.fixture
def setup_test_markdown_file(tmp_path: Path) -> Path:
    md_file = tmp_path / "test_document.md"
    md_file.write_text(TEST_MARKDOWN_CONTENT, encoding="utf-8")
    return md_file


@pytest.fixture
def setup_storage_env(tmp_path: Path):
    test_data_path = tmp_path / "test_file_storage"
    test_data_path.mkdir(parents=True, exist_ok=True)
    os.environ["FILE_STORAGE_PATH"] = str(test_data_path)
    yield test_data_path


@pytest.fixture
def vector_db_manager():
    manager = get_vector_db_manager()
    manager.create_collection(TEST_COLLECTION_NAME)
    yield manager
    manager.delete_collection(TEST_COLLECTION_NAME)


def test_graph_call(
    setup_test_markdown_file: Path,
    setup_storage_env: Path,
    vector_db_manager,
):
    md_file = setup_test_markdown_file
    chunks, sub_chunks = chunk_markdown(md_file)

    assert len(chunks) >= 1, "Should have at least one chunk"
    assert len(sub_chunks) >= 1, "Should have at least one sub_chunk"

    file_storage = FileStorageManager()
    chunk_ids: List[str] = []
    for i, chunk in enumerate(chunks):
        chunk_id: str = chunk.metadata.get(META_SOURCE_FILE_CHUNK_ID_KEY, "")
        chunk_ids.append(chunk_id)
        file_storage.save(
            id=chunk_id,
            content=chunk.page_content,
            metadata=chunk.metadata,
        )

    for chunk_id in chunk_ids:
        loaded = file_storage.load(chunk_id)
        assert loaded is not None, f"Chunk {chunk_id} should be loadable"
        assert "content" in loaded

    vector_store = vector_db_manager.get_collection(TEST_COLLECTION_NAME)
    vector_store.add_documents(sub_chunks)

    search_results = vector_store.similarity_search("mixed precision training", k=3)
    assert len(search_results) > 0, "Should find relevant documents"

    found_relevant = any(
        "precision" in result.page_content.lower()
        or "training" in result.page_content.lower()
        for result in search_results
    )
    assert found_relevant, "Search results should contain relevant content"

    llm = load_llm()
    tools = [retrieve_file_chunks, search_vector_chunk]
    graph = create_agent_graph(llm, tools)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    input_message = {
        "messages": [
            {
                "role": "user",
                "content": f"请使用 search_vector_chunk 工具，参数 collection_name 设置为 '{TEST_COLLECTION_NAME}'，搜索 'mixed precision training' 相关内容。",
            }
        ]
    }

    result = graph.invoke(input_message, config)

    assert result is not None, "Graph should return a result"
    assert "messages" in result, "Result should contain messages"
    assert len(result["messages"]) > 0, "Should have at least one message"

    print("\n" + "=" * 60)
    print("Graph Execution Result:")
    print("=" * 60)
    for msg in result["messages"]:
        print(f"[{type(msg).__name__}]: {getattr(msg, 'content', str(msg))[:500]}")
    print("=" * 60)


def test_hybrid_storage_search(
    setup_test_markdown_file: Path,
    setup_storage_env: Path,
    vector_db_manager: VectorDBManager,
):
    """Test hybrid storage: vector search sub_chunks -> locate parent chunks -> retrieve full content"""
    md_file = setup_test_markdown_file

    chunks, sub_chunks = chunk_markdown(md_file)
    file_storage = FileStorageManager()

    for i, chunk in enumerate(chunks):
        source_file = chunk.metadata.get(META_SOURCE_FILE_KEY)
        chunk_id = chunk.metadata[META_SOURCE_FILE_CHUNK_ID_KEY]
        assert isinstance(source_file, str) and isinstance(
            chunk_id, str
        ), "Source file and chunk id should be strings and integers"
        file_storage.save(
            id=chunk_id,
            content=chunk.page_content,
            metadata=chunk.metadata,
        )

    vector_store = vector_db_manager.get_collection(TEST_COLLECTION_NAME)
    vector_store.add_documents(sub_chunks)

    test_queries = [
        ("mixed precision training FP16 memory", "precision"),
        ("gradient checkpointing memory optimization", "checkpointing"),
        ("knowledge distillation teacher student", "distillation"),
    ]

    for query, expected_keyword in test_queries:

        sub_chunk_results = vector_store.similarity_search(query, k=3)
        assert len(sub_chunk_results) > 0, f"Should find sub_chunks for: {query}"

        chunk_indices: List[str] = [
            sub_result.metadata.get(META_SOURCE_FILE_CHUNK_ID_KEY, "")
            for sub_result in sub_chunk_results
        ]

        full_chunks = file_storage.load_batch(chunk_indices)
        assert len(full_chunks) > 0, "Should retrieve full chunks from file storage"

        combined_content = ""
        for full_chunk in full_chunks:
            chunk_content = full_chunk["content"]
            combined_content += chunk_content

        assert (
            expected_keyword.lower() in combined_content.lower()
        ), f"Full chunk should contain '{expected_keyword}'"

        print(f"query: {query}")
        print(f"combined_content: {combined_content}")
