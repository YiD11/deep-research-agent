from pathlib import Path

from util.doc import chunk_markdown


def _write_tmp_markdown(tmp_dir: Path) -> Path:
    content = """# Header 1

This is some text under header 1.

## Header 2

This is some text under header 2.

### Header 3

This is some text under header 3.
"""
    md_path = tmp_dir / "sample.md"
    md_path.write_text(content, encoding="utf-8")
    return md_path


def test_chunk_markdown_basic(tmp_path: Path):
    md_file = _write_tmp_markdown(tmp_path)

    chunks, sub_chunks = chunk_markdown(md_file)

    # 基本行为：应当返回至少一个头部 chunk 和至少一个子 chunk
    assert len(chunks) >= 1
    assert len(sub_chunks) >= 1

    # metadata 中应包含源文件名与 chunk id
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.get("source_file") == md_file.name
        assert chunk.metadata.get("chunk_id") == i


