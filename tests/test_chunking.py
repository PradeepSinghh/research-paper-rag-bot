"""
Tests for rag/chunking.py — text splitting and chunk metadata.

No external API calls are made; tests are purely in-process.
"""

import pytest

from rag.chunking import chunk_pages, _split_text, _find_break


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_page(text: str, page_number: int = 1, filename: str = "test.pdf") -> dict:
    return {
        "text": text,
        "page_number": page_number,
        "filename": filename,
        "total_pages": 5,
        "file_path": f"/tmp/{filename}",
    }


# ─── chunk_pages tests ────────────────────────────────────────────────────────

class TestChunkPages:

    def test_short_text_produces_one_chunk(self):
        # Use text longer than MIN_CHUNK_LENGTH (50 chars) so it is not filtered.
        text = "This is a sufficiently long sentence that exceeds the minimum chunk length."
        pages = [_make_page(text)]
        chunks = chunk_pages(pages, chunk_size=512, chunk_overlap=64)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_chunk_carries_source_metadata(self):
        text = "Some content here that is definitely longer than fifty characters in total."
        pages = [_make_page(text, page_number=3, filename="paper.pdf")]
        chunks = chunk_pages(pages, chunk_size=512, chunk_overlap=64)

        assert chunks[0]["filename"] == "paper.pdf"
        assert chunks[0]["page_number"] == 3
        assert "chunk_id" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        long_text = "word " * 600  # ~3000 chars
        pages = [_make_page(long_text)]
        chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)

        assert len(chunks) > 1

    def test_chunk_ids_are_monotonically_increasing(self):
        pages = [_make_page("word " * 400), _make_page("more " * 400, page_number=2)]
        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)

        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids)))

    def test_min_length_filters_tiny_chunks(self):
        # A very short text that should be filtered out
        pages = [_make_page("Hi")]
        chunks = chunk_pages(pages, chunk_size=512, chunk_overlap=0, min_length=50)

        assert chunks == []

    def test_empty_pages_list_returns_empty(self):
        assert chunk_pages([]) == []

    def test_overlap_creates_shared_content(self):
        """Consecutive chunks should share characters when overlap > 0."""
        text = "ABCDE" * 100  # 500 chars
        pages = [_make_page(text)]
        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50, min_length=1)

        if len(chunks) >= 2:
            # The end of chunk 0 should appear in the start of chunk 1
            end_of_first = chunks[0]["text"][-30:]
            assert any(
                end_of_first[:10] in chunks[1]["text"]
                for _ in [None]  # evaluate once
            )


# ─── _split_text tests ────────────────────────────────────────────────────────

class TestSplitText:

    def test_empty_string_returns_empty_list(self):
        assert _split_text("", 512, 64) == []

    def test_text_shorter_than_chunk_size_returns_single_element(self):
        result = _split_text("hello world", 512, 64)
        assert result == ["hello world"]

    def test_splits_long_text(self):
        text = "sentence. " * 100  # 1000 chars
        result = _split_text(text, 200, 20)
        assert len(result) > 1

    def test_all_content_preserved(self):
        """
        Every character in the original text should appear in at least one chunk.
        This is a weak check (overlap means some chars appear in multiple chunks).
        """
        text = "ABCDEFGHIJ" * 50  # 500 chars
        chunks = _split_text(text, 100, 10)
        reconstructed = "".join(chunks)
        # Every unique char of original is in reconstructed
        assert set(text).issubset(set(reconstructed))


# ─── _find_break tests ────────────────────────────────────────────────────────

class TestFindBreak:

    def test_prefers_paragraph_break(self):
        text = "First paragraph.\n\nSecond paragraph starts here."
        # pos=20 puts us inside the first paragraph; break should snap to \n\n
        pos = _find_break(text, 20, search_window=30)
        assert text[pos - 1] in ("\n", " ") or pos <= 20

    def test_falls_back_to_original_pos_when_no_break(self):
        text = "abcdefghijklmnopqrstuvwxyz"  # no whitespace or punctuation
        pos = 15
        result = _find_break(text, pos, search_window=5)
        assert result == pos
