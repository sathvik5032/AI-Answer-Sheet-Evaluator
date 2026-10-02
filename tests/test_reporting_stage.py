"""Test suite for Stage 8: Final Reporting and Packaging."""

import csv
import json
from pathlib import Path
import pytest

from src.reporter import ResultGenerator


@pytest.fixture(scope="module")
def final_reports():
    """Ensures final reports are generated and loads both JSON and CSV files."""
    json_path = Path("data/final/student_001_final_evaluation.json")
    csv_path = Path("data/final/student_001_final_evaluation.csv")

    if not json_path.exists() or not csv_path.exists():
        generator = ResultGenerator(output_dir="data/final")
        generator.generate_final_submission_reports()

    assert json_path.exists(), f"Final JSON report missing: {json_path}"
    assert csv_path.exists(), f"Final CSV report missing: {csv_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_records = list(reader)

    return {"json": json_data, "csv": csv_records}


def test_final_json_structure_and_scores(final_reports):
    """Verify final JSON report structure, total marks, and human review list."""
    data = final_reports["json"]
    assert data["student_id"] == "student_001"
    assert data["subject"] == "Computer Fundamentals"
    assert data["total_score"] == 20.0
    assert data["total_max_score"] == 25.0
    assert data["percentage"] == 80.0
    assert data["human_review_required"] is True
    assert data["human_review_questions"] == ["Q5"]

    summary = data["summary"]
    assert summary["total_questions"] == 5
    assert summary["high_confidence_count"] == 4
    assert summary["low_confidence_count"] == 1
    assert summary["human_review_required_count"] == 1


def test_final_json_every_question_has_all_required_fields(final_reports):
    """Verify every question in JSON has all assignment-required fields."""
    required_fields = {
        "question_id",
        "extracted_answer",
        "score",
        "max_score",
        "score_vs_key",
        "confidence",
        "reason",
        "human_review_required",
    }
    questions = final_reports["json"]["questions"]
    assert len(questions) == 5

    for q in questions:
        for rf in required_fields:
            assert rf in q, f"Missing required field '{rf}' in question {q.get('question_id')}"
        assert len(q["extracted_answer"].strip()) > 0
        assert len(q["score_vs_key"].strip()) > 0
        assert len(q["reason"].strip()) > 0
        assert q["max_score"] == 5.0


def test_final_csv_format_and_columns(final_reports):
    """Verify final CSV contains exact expected columns and 5 question rows."""
    records = final_reports["csv"]
    assert len(records) == 5

    expected_cols = [
        "question_id",
        "extracted_answer",
        "score",
        "max_score",
        "score_vs_key",
        "confidence",
        "reason",
        "human_review_required",
    ]
    for row in records:
        for col in expected_cols:
            assert col in row, f"Missing column '{col}' in CSV row {row.get('question_id')}"


def test_final_question_level_verdicts_and_human_review(final_reports):
    """Verify individual question scores, confidences, and human review flags."""
    expected = {
        "Q1": (5.0, "HIGH", False),
        "Q2": (5.0, "HIGH", False),
        "Q3": (5.0, "HIGH", False),
        "Q4": (0.0, "HIGH", False),
        "Q5": (5.0, "LOW", True),
    }

    # Verify JSON
    q_map_json = {q["question_id"]: q for q in final_reports["json"]["questions"]}
    for qid, (exp_score, exp_conf, exp_rev) in expected.items():
        q_item = q_map_json[qid]
        assert q_item["score"] == exp_score
        assert q_item["confidence"] == exp_conf
        assert q_item["human_review_required"] is exp_rev

    # Verify CSV
    q_map_csv = {r["question_id"]: r for r in final_reports["csv"]}
    for qid, (exp_score, exp_conf, exp_rev) in expected.items():
        row = q_map_csv[qid]
        assert float(row["score"]) == exp_score
        assert row["confidence"] == exp_conf
        assert (row["human_review_required"].lower() == "true") is exp_rev


def test_approach_note_exists_and_line_count():
    """Verify approach note exists and adheres to the 10-15 line constraint."""
    note_path = Path("data/final/approach_note.md")
    assert note_path.exists(), f"Approach note does not exist: {note_path}"

    with open(note_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 10 to 15 non-empty lines constraint
    assert 10 <= len(lines) <= 15, f"Approach note has {len(lines)} lines; expected 10 to 15 lines."


def test_final_readme_exists_and_covers_required_sections():
    """Verify data/final/README.md exists and documents all required aspects."""
    readme_path = Path("data/final/README.md")
    assert readme_path.exists(), f"README does not exist: {readme_path}"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Computer Fundamentals" in content
    assert "20.0" in content or "20" in content
    assert "Q5" in content
    assert "human review" in content.lower()
    assert "data/inputs/question_paper" in content
    assert "data/inputs/answer_sheets" in content
    assert "data/rubrics" in content
