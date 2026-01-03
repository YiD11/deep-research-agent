import json
import os
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.load import dumps
from const import META_SOURCE_FILE_CHUNK_ID_KEY


class FileStorageManager:
    _store_path: Path

    def __init__(self, store_path: str = None):
        store_path = store_path or os.getenv("FILE_STORAGE_PATH", "./data")
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)

    def save(self, id: str, content: str, metadata: dict) -> None:
        """Save a single document by ID."""
        file_path = self._store_path / f"{id}.json"
        data = {"id": id, "content": content, "metadata": metadata}
        with open(file_path, "w+", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def load(self, id: str) -> dict | None:
        """Load a single document by ID."""
        file_path = self._store_path / (id if id.endswith(".json") else f"{id}.json")
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def save_batch(self, batch: List[Document]) -> None:
        for doc in batch:
            file_path = (
                self._store_path / f"{doc.metadata[META_SOURCE_FILE_CHUNK_ID_KEY]}.json"
            )
            s = doc.model_dump_json(indent=2, ensure_ascii=False)
            with open(file_path, "w+", encoding="utf-8") as fp:
                fp.write(s)

    def load_batch(self, paths: List[str]) -> List[Document]:
        paths = set(paths)
        files_paths = [
            self._store_path / (p if p.endswith(".json") else f"{p}.json")
            for p in paths
        ]
        docs = []
        for p in files_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as fp:
                    s = fp.read()
                doc = Document.model_validate_json(s)
                docs.append(doc)
        return docs

    def clean_store(self) -> None:
        if self._store_path.exists():
            shutil.rmtree(self._store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
