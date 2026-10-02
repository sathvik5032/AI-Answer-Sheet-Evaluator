"""Test suite for Stage 5: Question Segmentation."""

import json
from pathlib import Path
import pytest

from src.models import (
    OCRSubmissionResult,
    SegmentedAnswer,
    SegmentedSubmissionResult,
)
from src.segmentation import QuestionSegmenter


@pytest.fixture(scope="module")
def segmentation_data():
    """Ensures segmentation has run and loads the resulting structured JSON output."""
    output_path = Path("data/ocr_extracted/student_001_segmented.json")
    if not output_path.exists():
        from scripts.run_segmentation import main
        main()

    assert output_path.exists(), f"Segmented output file does not exist: {output_path}"
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SegmentedSubmissionResult(**data)


def test_exactly_five_question_segments(segmentation_data):
    """Verify that exactly five question segments are produced (Q1 - Q5)."""
    assert segmentation_data.total_answers == 5
    assert len(segmentation_data.answers) == 5

    question_ids = [a.question_id for a in segmentation_data.answers]
    assert question_ids == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_q1_source_page(segmentation_data):
    """Verify Q1 belongs strictly to page1.jpg."""
    q1 = next(a for a in segmentation_data.answers if a.question_id == "Q1")
    assert q1.source_pages == ["page1.jpg"]
    assert q1.ocr_item_count > 0
    assert len(q1.text.strip()) > 0
    assert "computer" in q1.text.lower()


def test_q2_source_page(segmentation_data):
    """Verify Q2 belongs strictly to page1.jpg."""
    q2 = next(a for a in segmentation_data.answers if a.question_id == "Q2")
    assert q2.source_pages == ["page1.jpg"]
    assert q2.ocr_item_count > 0
    assert len(q2.text.strip()) > 0
    assert "ram" in q2.text.lower() or "memosy" in q2.text.lower()


def test_q3_cross_page_continuation(segmentation_data):
    """Verify Q3 contains OCR content from BOTH page1.jpg and page2.jpg."""
    q3 = next(a for a in segmentation_data.answers if a.question_id == "Q3")
    assert "page1.jpg" in q3.source_pages
    assert "page2.jpg" in q3.source_pages
    assert len(q3.source_pages) == 2

    # Check that items actually come from both pages
    item_pages = {item.page_number for item in q3.items}
    assert item_pages == {1, 2}

    # Verify key tokens spanning across page break
    text_lower = q3.text.lower()
    # Operating system portion on page 1
    assert "operating" in text_lower or "so/tware" in text_lower or "compube" in text_lower
    # Hardware / programs portion on page 2
    assert "hardware" in text_lower or "psograms" in text_lower or "uindows" in text_lower


def test_q4_source_page(segmentation_data):
    """Verify Q4 begins on page2.jpg after Q3 continuation and belongs to page2.jpg."""
    q4 = next(a for a in segmentation_data.answers if a.question_id == "Q4")
    assert q4.source_pages == ["page2.jpg"]
    assert q4.ocr_item_count > 0
    assert len(q4.text.strip()) > 0
    assert "ketboard" in q4.text.lower() or "output" in q4.text.lower()


def test_q5_source_page(segmentation_data):
    """Verify Q5 follows Q4 on page2.jpg."""
    q5 = next(a for a in segmentation_data.answers if a.question_id == "Q5")
    assert q5.source_pages == ["page2.jpg"]
    assert q5.ocr_item_count > 0
    assert len(q5.text.strip()) > 0
    assert "sottwaxe" in q5.text.lower() or "brouser" in q5.text.lower() or "websi" in q5.text.lower()


def test_all_answers_non_empty(segmentation_data):
    """Verify every segmented answer has non-empty text and non-zero confidence."""
    for ans in segmentation_data.answers:
        assert isinstance(ans.text, str)
        assert len(ans.text.strip()) > 0
        assert ans.ocr_item_count > 0
        assert 0.0 <= ans.average_confidence <= 1.0


def test_ocr_item_references_preserved(segmentation_data):
    """Verify detailed item references, bboxes, and orders are preserved."""
    total_preserved_items = 0
    for ans in segmentation_data.answers:
        assert len(ans.items) == ans.ocr_item_count
        total_preserved_items += len(ans.items)
        for item in ans.items:
            assert item.order >= 1
            assert item.page_number in [1, 2]
            assert item.source_file in ["page1.jpg", "page2.jpg"]
            assert len(item.text) > 0
            assert 0.0 <= item.confidence <= 1.0
            assert item.bounding_box is not None
            assert item.bounding_box.x_min >= 0
            assert item.bounding_box.y_min >= 0

    # Ensure all 135 OCR items are accounted for across all 5 answers
    assert total_preserved_items == 135
