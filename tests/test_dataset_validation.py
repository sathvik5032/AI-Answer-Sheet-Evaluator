"""Validation tests for the actual assignment dataset and schemas."""

import json
import os
import fitz
from PIL import Image
import cv2
import pytest

from src.models import MasterRubric


def test_actual_files_present():
    """Verify all five expected dataset files are present and non-empty."""
    expected_files = [
        "data/inputs/question_paper/Computer_Fundamentals_Question_Paper.pdf",
        "data/inputs/answer_sheets/page1.jpg",
        "data/inputs/answer_sheets/page2.jpg",
        "data/rubrics/computer_fundamentals.json",
        "data/references/ground_truth.json",
    ]
    for path in expected_files:
        assert os.path.isfile(path), f"Missing file: {path}"
        assert os.path.getsize(path) > 0, f"Empty file: {path}"


def test_question_paper_content():
    """Parse question paper and verify Q1-Q5, 5 questions, 25 total marks."""
    pdf_path = "data/inputs/question_paper/Computer_Fundamentals_Question_Paper.pdf"
    doc = fitz.open(pdf_path)
    assert len(doc) == 1, "Question paper should be 1 page"
    text = doc[0].get_text()

    assert "Computer Fundamentals" in text
    assert "Total Marks: 25" in text
    assert "5 Questions" in text

    # Verify each of the 5 questions is present in text
    assert "1. What is a computer?" in text
    assert "2. What is RAM?" in text
    assert "3. What is an operating system?" in text
    assert "4. What is the purpose of a keyboard?" in text
    assert "5. What is a web browser?" in text


def test_rubric_schema_and_criteria():
    """Verify computer_fundamentals.json has Q1-Q5, 5 marks each, total 25 marks."""
    rubric_path = "data/rubrics/computer_fundamentals.json"
    with open(rubric_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    master = MasterRubric(**data)
    assert master.subject == "Computer Fundamentals"
    assert master.total_marks == 25
    assert len(master.questions) == 5

    expected_qids = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for i, q in enumerate(master.questions):
        assert q.question_id == expected_qids[i]
        assert q.total_marks == 5.0
        assert len(q.prompt) > 0
        assert len(q.model_answer) > 0
        assert len(q.concepts) >= 4
        assert len(q.criteria) >= 3

        # Criteria marks must sum to 5.0
        criteria_total = sum(c.marks for c in q.criteria)
        assert criteria_total == 5.0, f"{q.question_id} criteria sum {criteria_total} != 5.0"


def test_ground_truth_content():
    """Verify ground truth matches test case specifications and records Q3 cross-page link."""
    gt_path = "data/references/ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    assert gt["student_id"] == "student_001"
    assert gt["exam_subject"] == "Computer Fundamentals"
    assert gt["total_expected_marks"] == 20
    assert len(gt["answers"]) == 5

    answers_map = {a["question_id"]: a for a in gt["answers"]}

    # Q1: clearly correct, 5 marks, page1
    q1 = answers_map["Q1"]
    assert q1["expected_marks"] == 5
    assert q1["evaluation_case"] == "clearly_correct"
    assert q1["source_pages"] == ["page1.jpg"]

    # Q2: correct with different wording, 5 marks, page1
    q2 = answers_map["Q2"]
    assert q2["expected_marks"] == 5
    assert q2["evaluation_case"] == "correct_different_wording"
    assert q2["source_pages"] == ["page1.jpg"]

    # Q3: correct across multiple pages, 5 marks, spans page1.jpg and page2.jpg
    q3 = answers_map["Q3"]
    assert q3["expected_marks"] == 5
    assert q3["evaluation_case"] == "correct_across_multiple_pages"
    assert q3["source_pages"] == ["page1.jpg", "page2.jpg"]

    # Q4: incorrect meaning with keywords, 0 marks, page2
    q4 = answers_map["Q4"]
    assert q4["expected_marks"] == 0
    assert q4["evaluation_case"] == "incorrect_meaning_with_keywords"
    assert q4["source_pages"] == ["page2.jpg"]

    # Q5: correct with extra information, 5 marks, page2
    q5 = answers_map["Q5"]
    assert q5["expected_marks"] == 5
    assert q5["evaluation_case"] == "correct_with_extra_information"
    assert q5["source_pages"] == ["page2.jpg"]


def test_images_validity():
    """Verify that page1.jpg and page2.jpg can be opened as valid images."""
    for img_name in ["page1.jpg", "page2.jpg"]:
        path = os.path.join("data/inputs/answer_sheets", img_name)
        # PIL check
        with Image.open(path) as img:
            assert img.format in ["JPEG", "JPG", "PNG"]
            assert img.size[0] > 100
            assert img.size[1] > 100

        # OpenCV check
        cv_img = cv2.imread(path)
        assert cv_img is not None, f"cv2 failed to load {path}"
        assert cv_img.shape[0] > 100
        assert cv_img.shape[1] > 100
