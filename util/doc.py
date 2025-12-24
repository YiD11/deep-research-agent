from pathlib import Path
from typing import List
import uuid
from langchain_core.documents.base import Document
import pymupdf
import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from const.document import *

def pdf_to_markdown(pdf_path: Path, markdown_path: Path):
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    pdf = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(
        pdf,
        ignore_images=True,
    )
    md_cleaned = md.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="ignore"
    )
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(md_cleaned)


def split_large_chunk(chunks: List[Document], max_size: int = MAX_CHUNK_SIZE):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=200,
    )
    res: List[Document] = []
    for chunk in chunks:
        if len(chunk.page_content) > max_size:
            sub_chunks = splitter.split_documents([chunk])
            res.extend(sub_chunks)
        else:
            res.append(chunk)
    return res


def merge_small_chunk(chunks: List[Document], min_size: int = MIN_CHUNK_SIZE):
    merged_chunks: List[Document] = []
    buffer_content = ""
    buffer_metadata = {}
    for chunk in chunks:
        if len(buffer_content) + len(chunk.page_content) < min_size:
            buffer_content += "\n" + chunk.page_content
        else:
            if buffer_content:
                merged_chunks.append(
                    Document(
                        page_content=buffer_content.strip(), metadata=buffer_metadata
                    )
                )
            buffer_content = chunk.page_content
            buffer_metadata = chunk.metadata
    if buffer_content:
        merged_chunks.append(
            Document(page_content=buffer_content.strip(), metadata=buffer_metadata)
        )
    return merged_chunks


def chunk_markdown(file: Path, sub_chunk_size: int = SUB_MAX_CHUNK_SIZE):
    if not file.is_file():
        raise FileNotFoundError(f"Markdown file not found: {file}")
    header_spliter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    sub_chunker = RecursiveCharacterTextSplitter(
        chunk_size=sub_chunk_size,
        chunk_overlap=200,
    )
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
    head_chunks: List[Document] = header_spliter.split_text(content)
    merged_chunks = merge_small_chunk(head_chunks)
    splitted_chunks = split_large_chunk(merged_chunks)
    all_sub_chunks = []
    for i, chunk in enumerate(splitted_chunks):
        chunk.metadata[META_SOURCE_FILE_KEY] = file.name
        chunk.metadata[META_SOURCE_FILE_CHUNK_ID_KEY] = uuid.uuid4().hex
        sub_chunks = sub_chunker.split_documents([chunk])
        # sub_chunk also contains the same meta data as 
        all_sub_chunks.extend(sub_chunks)
    return splitted_chunks, all_sub_chunks
    