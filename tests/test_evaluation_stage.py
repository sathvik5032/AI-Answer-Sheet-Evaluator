"""Test suite for Stage 6: Semantic Answer Evaluation."""

import json
from pathlib import Path
import pytest

from src.evaluator import SemanticEvaluator
from src.models import (
    CriterionEvaluation,
    MasterRubric,
    QuestionEvaluation,
    RubricCriterionItem,
    SegmentedAnswer,
    SegmentedSubmissionResult,
    StudentEvaluationResult,
)


@pytest.fixture(scope="module")
def evaluation_data():
    """Ensures evaluation has run and loads the resulting structured JSON output."""
    output_path = Path("data/results/student_001_evaluation.json")
    if not output_path.exists():
        from scripts.run_evaluation import main
        main()

    assert output_path.exists(), f"Evaluation output file does not exist: {output_path}"
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return StudentEvaluationResult(**data)


def test_all_five_questions_evaluated(evaluation_data):
    """Verify that all five questions (Q1 - Q5) are evaluated."""
    assert len(evaluation_data.questions) == 5
    qids = [q.question_id for q in evaluation_data.questions]
    assert qids == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_scores_within_valid_rubric_limits(evaluation_data):
    """Verify that every question has a score and all scores are within [0, max_score]."""
    total = 0.0
    for qe in evaluation_data.questions:
        assert isinstance(qe.score, (int, float))
        assert isinstance(qe.max_score, (int, float))
        assert 0.0 <= qe.score <= qe.max_score
        assert qe.max_score == 5.0
        total += qe.score

    assert evaluation_data.total_score == total
    assert evaluation_data.total_score <= 25.0
    assert evaluation_data.total_score == 20.0


def test_q1_evaluates_correctly(evaluation_data):
    """Verify Q1 receives full credit for clearly correct definitions."""
    q1 = next(q for q in evaluation_data.questions if q.question_id == "Q1")
    assert q1.score == 5.0
    assert len(q1.criterion_evaluations) == 5
    for ce in q1.criterion_evaluations:
        assert ce.status == "satisfied"
        assert ce.marks_awarded > 0


def test_q2_different_wording_and_ocr_noise(evaluation_data):
    """Verify Q2 receives full credit despite different wording and OCR noise."""
    q2 = next(q for q in evaluation_data.questions if q.question_id == "Q2")
    assert q2.score == 5.0
    assert len(q2.criterion_evaluations) == 4
    for ce in q2.criterion_evaluations:
        assert ce.status == "satisfied"
        assert ce.marks_awarded > 0


def test_q3_cross_page_answer_evaluation(evaluation_data):
    """Verify Q3 evaluates the complete cross-page answer as a unified whole."""
    q3 = next(q for q in evaluation_data.questions if q.question_id == "Q3")
    assert q3.score == 5.0
    assert len(q3.criterion_evaluations) == 5

    # Page 1 concepts (software, user interaction) and Page 2 concepts (hardware, run programs, Windows)
    statuses = [ce.status for ce in q3.criterion_evaluations]
    assert all(s == "satisfied" for s in statuses)


def test_q4_not_awarded_keyword_overlap_and_detects_contradiction(evaluation_data):
    """Critical Test: Q4 must NOT be awarded marks merely because of keyword overlap,

    and must recognize the factual contradiction between input and output device.
    """
    q4 = next(q for q in evaluation_data.questions if q.question_id == "Q4")
    # Score must be 0
    assert q4.score == 0.0

    # Must contain contradiction status in criterion evaluations
    contradicted_criteria = [ce for ce in q4.criterion_evaluations if ce.status == "contradicted"]
    assert len(contradicted_criteria) >= 1

    # Check reason explicitly mentions the contradiction
    reasons_text = " ".join(ce.reason for ce in q4.criterion_evaluations)
    assert "output device" in reasons_text.lower() or "contradiction" in reasons_text.lower()


def test_q5_handles_extra_information_without_penalty(evaluation_data):
    """Verify Q5 handles extra information without incorrectly penalizing the score."""
    q5 = next(q for q in evaluation_data.questions if q.question_id == "Q5")
    assert q5.score == 5.0
    assert len(q5.criterion_evaluations) == 4
    for ce in q5.criterion_evaluations:
        assert ce.status == "satisfied"
        assert ce.marks_awarded > 0


def test_no_keyword_only_scoring_logic():
    """Verify that a sentence with high keyword overlap but contradictory meaning is NOT scored purely by keywords."""
    evaluator = SemanticEvaluator()

    # Synthetic negative test: text has relevant keywords (keyboard, input, device, computer, enter) but negates it
    contradictory_text = "A keyboard is definitely NOT an input device. It is an output device that displays on the screen."
    norm_text = evaluator.normalize_ocr_text(contradictory_text)
    contra = evaluator.detect_contradiction("Q4", norm_text, "Correctly identifies a keyboard as an input device")
    assert contra is not None
    assert "contradiction" in contra.lower()


def test_synthetic_contradiction_detection_generalized_without_q4():
    """Test A: Contradiction can be detected without question_id == 'Q4'.

    Demonstrates that any arbitrary question ID with contradictory opposing classifications
    (e.g., input vs output device, volatile vs permanent memory) triggers contradiction detection.
    """
    evaluator = SemanticEvaluator()

    # Case 1: Arbitrary Question ID "QUESTION_99" with input vs output device
    student_answer = "A mouse is an output device that displays graphics on the screen."
    norm_text = evaluator.normalize_ocr_text(student_answer)
    criterion_text = "Correctly identifies mouse as an input device for entering commands."

    contra = evaluator.detect_contradiction(norm_text, criterion_text)
    assert contra is not None
    assert "contradiction" in contra.lower()
    assert "output device" in contra.lower()

    # Evaluate full criterion using an arbitrary question_id
    crit_item = RubricCriterionItem(
        criterion_id="Q99_C1",
        description="Identifies as an input device",
        marks=2.0,
        text="Identifies device as an input device",
    )
    result = evaluator.evaluate_criterion(
        criterion_id="Q99_C1",
        criterion=crit_item,
        expected_concept="input device",
        student_raw_text=student_answer,
        student_norm_text=norm_text,
        question_id="QUESTION_99",
    )
    assert result.status == "contradicted"
    assert result.marks_awarded == 0.0

    # Case 2: Volatile vs Permanent memory opposition
    student_ram = "RAM is a permanent non-volatile storage where data is never lost."
    ram_norm = evaluator.normalize_ocr_text(student_ram)
    ram_crit = "Explains that RAM is temporary volatile memory lost when switched off."
    contra_ram = evaluator.detect_contradiction(ram_norm, ram_crit)
    assert contra_ram is not None
    assert "volatile" in contra_ram.lower() or "permanent" in contra_ram.lower()


def test_synthetic_example_validation_generalized_without_q1_q3_q5():
    """Test B: Example criterion can be evaluated without question_id == Q1, Q3 or Q5.

    Demonstrates that arbitrary questions with example requirements correctly resolve
    named entities or domain categories without question_id heuristics.
    """
    evaluator = SemanticEvaluator()

    # Case 1: Web browser example under arbitrary ID "Q_BROWSER_42"
    student_text = "I browse websites using Google Chrome every day."
    criterion_text = "Provides a valid example of a web browser such as Chrome or Firefox."
    is_valid = evaluator.check_example_criterion(student_text, criterion_text)
    assert is_valid is True

    # Case 2: Operating system example via category mapping under arbitrary ID "Q_OS_88"
    student_os = "My laptop runs Ubuntu Linux."
    criterion_os = "Names an example operating system."
    is_valid_os = evaluator.check_example_criterion(student_os, criterion_os)
    assert is_valid_os is True

    # Case 3: Negative test - student gives no example or wrong category
    student_fail = "An operating system manages hardware components."
    is_valid_fail = evaluator.check_example_criterion(student_fail, criterion_os)
    assert is_valid_fail is False


def test_synthetic_ocr_normalization_independent_of_student_id():
    """Test C: OCR normalization does not depend on student_id.

    Verifies that text normalization is purely algorithmic (character confusion
    rules + domain vocabulary fuzzy matching) and takes no student_id argument.
    """
    evaluator = SemanticEvaluator()
    import inspect

    # Verify signature has no student_id parameter
    sig = inspect.signature(evaluator.normalize_ocr_text)
    assert "student_id" not in sig.parameters

    # Verify fuzzy vocabulary recovery works generally across multiple noisy words
    noisy_input = "Swf tched psograms on computex with ketboard and memosy"
    normalized = evaluator.normalize_ocr_text(noisy_input)

    assert "switched" in normalized.lower()
    assert "programs" in normalized.lower()
    assert "computer" in normalized.lower()
    assert "keyboard" in normalized.lower()
    assert "memory" in normalized.lower()

