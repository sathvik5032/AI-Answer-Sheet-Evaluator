"""Test suite for Stage 4: Handwriting OCR Extraction."""

import json
import os
import pytest
from pathlib import Path

from src.models import (
    OCRItem,
    OCRPageResult,
    OCRSubmissionResult,
    TextBoundingBox,
)
from src.ocr_engine import EasyOCREngine
from src.preprocessing import DocumentPreprocessor


@pytest.fixture(scope="module")
def ocr_submission_data():
    """Ensures OCR has run and loads the resulting structured JSON output."""
    output_path = Path("data/ocr_extracted/student_001_ocr.json")
    if not output_path.exists():
        # Execute extraction if not already run
        from scripts.run_ocr_pipeline import main
        main()

    assert output_path.exists(), f"OCR output file does not exist: {output_path}"
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return OCRSubmissionResult(**data)


def test_both_images_processed(ocr_submission_data):
    """Verify that both answer sheet pages were processed under one student submission."""
    assert ocr_submission_data.student_id == "student_001"
    assert ocr_submission_data.total_pages == 2
    assert len(ocr_submission_data.pages) == 2

    source_files = [p.source_file for p in ocr_submission_data.pages]
    assert "page1.jpg" in source_files
    assert "page2.jpg" in source_files


def test_page_numbers_preserved(ocr_submission_data):
    """Verify that page numbers are sequentially preserved and distinguishable."""
    p1 = ocr_submission_data.pages[0]
    p2 = ocr_submission_data.pages[1]

    assert p1.page_number == 1
    assert p1.source_file == "page1.jpg"

    assert p2.page_number == 2
    assert p2.source_file == "page2.jpg"

    # Both pages must have independent non-empty items
    assert p1.items_count > 0
    assert p2.items_count > 0
    assert p1.items_count == len(p1.items)
    assert p2.items_count == len(p2.items)

    # Page texts should be distinct
    assert p1.page_text != p2.page_text
    assert len(p1.page_text) > 50
    assert len(p2.page_text) > 50


def test_ocr_items_have_required_fields(ocr_submission_data):
    """Verify every OCR item has valid text, confidence, and bounding box."""
    for page in ocr_submission_data.pages:
        for item in page.items:
            # 1. Text check
            assert isinstance(item.text, str)
            assert len(item.text.strip()) > 0

            # 2. Confidence check
            assert isinstance(item.confidence, float)
            assert 0.0 <= item.confidence <= 1.0

            # 3. Polygon Bbox check
            assert isinstance(item.bbox, list)
            assert len(item.bbox) == 4
            for pt in item.bbox:
                assert len(pt) == 2
                assert isinstance(pt[0], int)
                assert isinstance(pt[1], int)

            # 4. TextBoundingBox check
            assert item.bounding_box is not None
            assert item.bounding_box.x_min >= 0
            assert item.bounding_box.y_min >= 0
            assert item.bounding_box.x_max >= item.bounding_box.x_min
            assert item.bounding_box.y_max >= item.bounding_box.y_min

            # 5. Order and page number
            assert item.order >= 1
            assert item.page_number == page.page_number


def test_preprocessor_loading():
    """Verify DocumentPreprocessor can load both images without corrupting dimensions."""
    prep = DocumentPreprocessor()
    for fname in ["page1.jpg", "page2.jpg"]:
        path = os.path.join("data/inputs/answer_sheets", fname)
        imgs = prep.load_document(path)
        assert len(imgs) == 1
        img = imgs[0]
        assert img.shape == (1280, 960, 3)

        # Preprocess verification
        proc, meta = prep.preprocess_image(img)
        assert proc.shape[:2] == (1280, 960)
        assert "deskew_angle" in meta
        assert "contrast_enhanced" in meta
