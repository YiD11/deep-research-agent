from dataclasses import dataclass
import json
import os
from typing import List, Dict, Optional, TypedDict
from sqlalchemy import create_engine, Column, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager


class Document(TypedDict):
    id: str
    content: str
    metadata: Dict


Base = declarative_base()


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String(255), primary_key=True, index=True)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=False, default={})


class PostgresStorageManager:
    _engine = None
    _session_factory = None

    def __init__(self):
        # Read database configuration from environment variables
        db_user = os.getenv("POSTGRES_USER", "deepresearch")
        db_password = os.getenv("POSTGRES_PASSWORD", "deepresearch123")
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "deepresearch")

        # Create database connection URL
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        # Create engine and session factory
        self._engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,
        )

        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self._engine)

    @contextmanager
    def _get_session(self) -> Session:
        """Context manager for database sessions"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save(self, id: str, content: str, metadata: Dict) -> None:
        """Save a single document to PostgreSQL"""
        with self._get_session() as session:
            # Check if document exists
            existing_doc = session.query(DocumentModel).filter_by(id=id).first()

            if existing_doc:
                # Update existing document
                existing_doc.content = content
                existing_doc.metadata = metadata
            else:
                # Create new document
                new_doc = DocumentModel(
                    id=id,
                    content=content,
                    metadata=metadata,
                )
                session.add(new_doc)

    def save_batch(self, batch: List[Document]) -> None:
        """Save multiple documents to PostgreSQL"""
        with self._get_session() as session:
            for doc in batch:
                existing_doc = (
                    session.query(DocumentModel).filter_by(id=doc["id"]).first()
                )

                if existing_doc:
                    existing_doc.content = doc["content"]
                    existing_doc.metadata = doc["metadata"]
                else:
                    new_doc = DocumentModel(
                        id=doc["id"],
                        content=doc["content"],
                        metadata=doc["metadata"],
                    )
                    session.add(new_doc)

    def load(self, id: str) -> Dict:
        """Load a single document from PostgreSQL"""
        with self._get_session() as session:
            doc = session.query(DocumentModel).filter_by(id=id).first()

            if not doc:
                raise FileNotFoundError(f"Document with id '{id}' not found")

            return {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
            }

    def load_batch(self, ids: List[str]) -> List[Dict]:
        """Load multiple documents from PostgreSQL"""
        unique_ids = sorted(set(ids))
        results = []

        with self._get_session() as session:
            for id in unique_ids:
                doc = session.query(DocumentModel).filter_by(id=id).first()

                if doc:
                    results.append(
                        {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata,
                        }
                    )

        return results

    def exists(self, id: str) -> bool:
        """Check if a document exists in PostgreSQL"""
        with self._get_session() as session:
            return session.query(DocumentModel).filter_by(id=id).count() > 0

    def delete(self, id: str) -> None:
        """Delete a single document from PostgreSQL"""
        with self._get_session() as session:
            doc = session.query(DocumentModel).filter_by(id=id).first()

            if doc:
                session.delete(doc)

    def delete_batch(self, ids: List[str]) -> None:
        """Delete multiple documents from PostgreSQL"""
        with self._get_session() as session:
            session.query(DocumentModel).filter(DocumentModel.id.in_(ids)).delete(
                synchronize_session=False
            )

    def list_all(self) -> List[str]:
        """List all document IDs in PostgreSQL"""
        with self._get_session() as session:
            docs = session.query(DocumentModel.id).all()
            return [doc.id for doc in docs]

    def count(self) -> int:
        """Count total documents in PostgreSQL"""
        with self._get_session() as session:
            return session.query(DocumentModel).count()

    def clear_store(self) -> None:
        """Clear all documents from PostgreSQL"""
        with self._get_session() as session:
            session.query(DocumentModel).delete()

    def close(self) -> None:
        """Close database connections"""
        if self._engine:
            self._engine.dispose()


_postgres_storage_manager = None


def get_postgres_storage_manager() -> PostgresStorageManager:
    """Get or create a singleton instance of PostgresStorageManager"""
    global _postgres_storage_manager
    if _postgres_storage_manager is None:
        _postgres_storage_manager = PostgresStorageManager()
    return _postgres_storage_manager
