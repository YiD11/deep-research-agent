import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from const import SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)


class VectorDBManager:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", "6333"))
        api_key = api_key or os.getenv("QDRANT_API_KEY", "secret")
        self.cli = QdrantClient(
            url=f"http://{host}:{port}",
            api_key=api_key,
        )
        dense_model_name = os.getenv("QDRANT_DENSE_EMBEDDING_MODEL_NAME", "")
        sparse_model_name = os.getenv("QDRANT_SPARSE_EMBEDDING_MODEL_NAME", "")
        logger.info(
            f"initializing vector db with dense model {dense_model_name} and sparse model {sparse_model_name}"
        )
        self._dense_embedding = HuggingFaceEmbeddings(model_name=dense_model_name)
        self._sparse_embedding = FastEmbedSparse(model_name=sparse_model_name)

    def create_collection(
        self,
        name: str,
    ) -> bool:
        if self.cli.collection_exists(name):
            return False
        self.cli.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=len(self._dense_embedding.embed_query("test")),
                distance=qmodels.Distance.COSINE,
            ),
            sparse_vectors_config={SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
        )
        return True

    def delete_collection(
        self,
        name: str,
    ) -> None:
        if self.cli.collection_exists(name):
            self.cli.delete_collection(name)

    def get_collection(self, collection_name: str) -> QdrantVectorStore:
        _ = self.create_collection(collection_name)
        return QdrantVectorStore(
            client=self.cli,
            collection_name=collection_name,
            embedding=self._dense_embedding,
            sparse_embedding=self._sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )


_vector_db_manager = None


def get_vector_db_manager() -> VectorDBManager:
    global _vector_db_manager
    if _vector_db_manager is None:
        _vector_db_manager = VectorDBManager()
    return _vector_db_manager
