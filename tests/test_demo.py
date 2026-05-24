"""
Tests for the sample/demo mode utilities.
"""

import os
import pytest
from rag.demo import (
    ensure_sample_pdf,
    get_sample_pdf_bytes,
    SAMPLE_FILENAME,
    SAMPLE_QUESTIONS,
    SAMPLE_PATH,
    SAMPLE_DIR,
)


class TestSampleDemo:
    def test_sample_filename_is_pdf(self):
        assert SAMPLE_FILENAME.endswith(".pdf")

    def test_sample_questions_not_empty(self):
        assert len(SAMPLE_QUESTIONS) >= 3

    def test_sample_questions_are_strings(self):
        for q in SAMPLE_QUESTIONS:
            assert isinstance(q, str)
            assert len(q) > 10

    def test_ensure_sample_pdf_returns_path(self):
        path = ensure_sample_pdf()
        assert os.path.exists(path)
        assert path.endswith(".pdf")

    def test_sample_pdf_is_nonzero(self):
        ensure_sample_pdf()
        size = os.path.getsize(SAMPLE_PATH)
        assert size > 1000, f"Sample PDF too small: {size} bytes"

    def test_get_sample_pdf_bytes_returns_bytes(self):
        data = get_sample_pdf_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 1000

    def test_pdf_bytes_start_with_pdf_magic(self):
        data = get_sample_pdf_bytes()
        assert data[:4] == b"%PDF", "Expected PDF magic bytes"

    def test_idempotent_ensure(self):
        """Calling ensure_sample_pdf twice should not raise."""
        p1 = ensure_sample_pdf()
        p2 = ensure_sample_pdf()
        assert p1 == p2

    def test_sample_pdf_loadable(self):
        """The sample PDF must be openable by PyMuPDF."""
        import fitz
        path = ensure_sample_pdf()
        doc = fitz.open(path)
        assert len(doc) >= 1
        doc.close()
