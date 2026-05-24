"""
Tests for rag/loader.py — PDF loading and text extraction.

A minimal in-memory PDF is generated with PyMuPDF for each test so the
suite runs without needing pre-made fixture files.
"""

import os
import tempfile

import fitz  # PyMuPDF
import pytest

from rag.loader import load_pdf


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_pdf(path: str, pages: list[str]) -> str:
    """
    Create a minimal single- or multi-page PDF at *path*.

    Args:
        path:  Destination file path (must end in .pdf).
        pages: List of text strings, one per page.

    Returns:
        *path* unchanged (for convenient use in with-statements).
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 100), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestLoadPdf:

    def test_single_page_returns_one_page_dict(self, tmp_path):
        pdf_path = str(tmp_path / "single.pdf")
        _make_pdf(pdf_path, ["Hello research world!"])

        pages = load_pdf(pdf_path)

        assert len(pages) == 1
        assert "Hello research world!" in pages[0]["text"]
        assert pages[0]["page_number"] == 1
        assert pages[0]["filename"] == "single.pdf"
        assert pages[0]["total_pages"] == 1

    def test_multi_page_returns_correct_count(self, tmp_path):
        pdf_path = str(tmp_path / "multi.pdf")
        _make_pdf(pdf_path, ["Page one content.", "Page two content.", "Page three."])

        pages = load_pdf(pdf_path)

        assert len(pages) == 3
        assert pages[0]["page_number"] == 1
        assert pages[1]["page_number"] == 2
        assert pages[2]["page_number"] == 3
        assert pages[0]["total_pages"] == 3

    def test_page_text_is_cleaned(self, tmp_path):
        pdf_path = str(tmp_path / "clean.pdf")
        _make_pdf(pdf_path, ["  Leading and trailing spaces.  "])

        pages = load_pdf(pdf_path)

        assert pages[0]["text"] == pages[0]["text"].strip()

    def test_metadata_fields_present(self, tmp_path):
        pdf_path = str(tmp_path / "meta.pdf")
        _make_pdf(pdf_path, ["Some content."])

        pages = load_pdf(pdf_path)

        required_keys = {"text", "page_number", "filename", "total_pages", "file_path"}
        assert required_keys.issubset(pages[0].keys())

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pdf("/non/existent/path/paper.pdf")

    def test_invalid_file_raises_value_error(self, tmp_path):
        bad_file = tmp_path / "not_a_pdf.pdf"
        bad_file.write_bytes(b"this is not a PDF")

        with pytest.raises(ValueError):
            load_pdf(str(bad_file))

    def test_empty_pdf_returns_empty_list(self, tmp_path):
        """A PDF with no text (blank page) should return an empty list."""
        pdf_path = str(tmp_path / "blank.pdf")
        doc = fitz.open()
        doc.new_page()          # blank page — no text inserted
        doc.save(pdf_path)
        doc.close()

        pages = load_pdf(pdf_path)

        assert pages == []
