"""
Tests for the reference / bibliography extraction module.
"""

import pytest
from rag.references import (
    extract_references,
    format_references_markdown,
    _find_ref_section_start,
    _split_into_entries,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _page(text: str, page_number: int = 1):
    return {
        "text": text,
        "page_number": page_number,
        "filename": "paper.pdf",
        "total_pages": 5,
    }


REF_SECTION_TEXT = (
    "References\n\n"
    "[1] Smith, J. et al. (2020). Attention is All You Need. NeurIPS.\n"
    "[2] Doe, A. (2021). Dense Passage Retrieval. EMNLP.\n"
    "[3] Brown, T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS.\n"
)

NUMBERED_DOT_TEXT = (
    "References\n\n"
    "1. Smith, J. (2020). Title One. Journal A.\n"
    "2. Jones, B. (2021). Title Two. Journal B.\n"
)


# ── _find_ref_section_start ───────────────────────────────────────────────────

class TestFindRefSectionStart:
    def test_finds_references_heading(self):
        pages = [
            _page("Introduction text here.", 1),
            _page("Methods section content.", 2),
            _page(REF_SECTION_TEXT, 3),
        ]
        idx = _find_ref_section_start(pages)
        assert idx == 2

    def test_finds_bibliography_heading(self):
        pages = [
            _page("Introduction here.", 1),
            _page("Bibliography\n\n[1] A. Author. Title. 2020.", 2),
        ]
        idx = _find_ref_section_start(pages)
        assert idx == 1

    def test_no_reference_section(self):
        pages = [_page("Just main text, no references.", 1)]
        assert _find_ref_section_start(pages) is None

    def test_empty_pages(self):
        assert _find_ref_section_start([]) is None

    def test_case_insensitive(self):
        pages = [_page("REFERENCES\n\n[1] Something.", 1)]
        assert _find_ref_section_start(pages) == 0


# ── _split_into_entries ───────────────────────────────────────────────────────

class TestSplitIntoEntries:
    def test_bracketed_numbers(self):
        text = (
            "[1] Smith, J. (2020). Title One.\n"
            "[2] Jones, B. (2021). Title Two.\n"
        )
        entries = _split_into_entries(text)
        assert len(entries) >= 2

    def test_dot_numbers(self):
        text = (
            "1. Smith, J. (2020). Title One.\n"
            "2. Jones, B. (2021). Title Two.\n"
        )
        entries = _split_into_entries(text)
        assert len(entries) >= 2

    def test_fallback_blank_lines(self):
        text = (
            "Smith, J. (2020). Title One. Journal A, vol 1.\n\n"
            "Jones, B. (2021). Title Two. Journal B, vol 2.\n"
        )
        entries = _split_into_entries(text)
        assert len(entries) >= 1


# ── extract_references ────────────────────────────────────────────────────────

class TestExtractReferences:
    def test_extracts_from_ref_section(self):
        pages = [
            _page("Main paper body text goes here.", 1),
            _page(REF_SECTION_TEXT, 2),
        ]
        refs = extract_references(pages, "paper.pdf")
        assert len(refs) >= 2

    def test_entry_has_required_keys(self):
        pages = [_page(REF_SECTION_TEXT, 1)]
        refs = extract_references(pages, "paper.pdf")
        assert refs, "Expected at least one reference"
        for ref in refs:
            assert "index" in ref
            assert "text" in ref
            assert "filename" in ref
            assert "page" in ref

    def test_filename_tagged(self):
        pages = [_page(REF_SECTION_TEXT, 1)]
        refs = extract_references(pages, "mypaper.pdf")
        assert all(r["filename"] == "mypaper.pdf" for r in refs)

    def test_no_ref_section(self):
        pages = [_page("Just introduction text, no references.", 1)]
        refs = extract_references(pages, "paper.pdf")
        assert refs == []

    def test_empty_pages(self):
        refs = extract_references([], "paper.pdf")
        assert refs == []

    def test_index_is_sequential(self):
        pages = [_page(REF_SECTION_TEXT, 1)]
        refs = extract_references(pages, "paper.pdf")
        if len(refs) >= 2:
            indices = [r["index"] for r in refs]
            assert indices == list(range(1, len(refs) + 1))

    def test_numbered_dot_format(self):
        pages = [_page(NUMBERED_DOT_TEXT, 1)]
        refs = extract_references(pages, "paper.pdf")
        assert len(refs) >= 1


# ── format_references_markdown ────────────────────────────────────────────────

class TestFormatReferencesMarkdown:
    def test_empty(self):
        assert format_references_markdown([]) == ""

    def test_numbered_output(self):
        refs = [
            {"index": 1, "text": "Smith, J. (2020). Title.", "filename": "p.pdf", "page": 5},
            {"index": 2, "text": "Jones, B. (2021). Other.", "filename": "p.pdf", "page": 5},
        ]
        md = format_references_markdown(refs)
        assert "1." in md
        assert "2." in md
        assert "Smith" in md
        assert "Jones" in md
