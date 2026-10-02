"""Test suite for Stage 7: Evaluation Confidence Scoring."""

import json
from pathlib import Path
import pytest

from src.confidence import ConfidenceScorer
from src.models import (
    CriterionEvaluation,
    QuestionConfidenceResult,
    QuestionEvaluation,
    StudentConfidenceResult,
    StudentEvaluationResult,
)


@pytest.fixture(scope="module")
def confidence_data():
    """Ensures confidence scoring has run and loads the resulting structured JSON output."""
    output_path = Path("data/results/student_001_confidence.json")
    if not output_path.exists():
        from scripts.run_confidence import main
        main()

    assert output_path.exists(), f"Confidence output file does not exist: {output_path}"
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return StudentConfidenceResult(**data)


@pytest.fixture(scope="module")
def stage6_evaluation_data():
    """Loads Stage 6 semantic evaluation data for score consistency checks."""
    eval_path = Path("data/results/student_001_evaluation.json")
    assert eval_path.exists(), f"Stage 6 evaluation file does not exist: {eval_path}"
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return StudentEvaluationResult(**data)


def test_confidence_output_contains_all_five_questions(confidence_data):
    """Verify that confidence output contains exactly 5 questions (Q1 - Q5)."""
    assert len(confidence_data.questions) == 5
    qids = [q.question_id for q in confidence_data.questions]
    assert qids == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_every_question_receives_valid_confidence_label(confidence_data):
    """Verify every question receives exactly one label from {HIGH, MEDIUM, LOW}."""
    valid_labels = {"HIGH", "MEDIUM", "LOW"}
    for qc in confidence_data.questions:
        assert qc.confidence.label in valid_labels
        assert isinstance(qc.confidence.label, str)


def test_every_question_has_non_empty_reason(confidence_data):
    """Verify every question has a meaningful, explanatory reason."""
    for qc in confidence_data.questions:
        reason = qc.confidence.reason.strip()
        assert len(reason) > 15
        # Ensure reasons are informative, not generic
        assert reason.lower() != "confidence is high."
        assert reason.lower() != "confidence is low."


def test_scores_exactly_match_stage6(confidence_data, stage6_evaluation_data):
    """Verify confidence output does not alter or recompute Stage 6 scores."""
    s6_scores = {q.question_id: (q.score, q.max_score) for q in stage6_evaluation_data.questions}
    for qc in confidence_data.questions:
        expected_score, expected_max = s6_scores[qc.question_id]
        assert qc.score == expected_score
        assert qc.max_score == expected_max


def test_q4_confidence_references_explicit_contradiction(confidence_data):
    """Verify Q4 has HIGH confidence grounded specifically in the factual contradiction."""
    q4 = next(q for q in confidence_data.questions if q.question_id == "Q4")
    assert q4.score == 0.0
    assert q4.confidence.label == "HIGH"
    reason = q4.confidence.reason.lower()
    assert "contradiction" in reason
    assert "output device" in reason


def test_human_review_consistency(confidence_data):
    """Verify LOW confidence requires human review, while HIGH/MEDIUM do not."""
    for qc in confidence_data.questions:
        if qc.confidence.label == "LOW":
            assert qc.human_review_required is True
            assert qc.question_id in confidence_data.human_review_questions
        else:
            assert qc.human_review_required is False
            assert qc.question_id not in confidence_data.human_review_questions


def test_confidence_not_simple_copy_of_ocr_confidence(confidence_data):
    """Verify evaluation confidence is a distinct multi-factor semantic assessment,

    not a duplicate of OCR confidence.
    """
    for qc in confidence_data.questions:
        factors = qc.confidence.factors
        assert "ocr_confidence" in factors
        assert "average_semantic_similarity" in factors
        # Q4 has non-zero OCR confidence (~0.667), but semantic similarity is 0 due to contradiction
        if qc.question_id == "Q4":
            assert factors["ocr_confidence"] > 0.60
            assert factors["has_contradiction"] is True


# =========================================================================
# SYNTHETIC EDGE-CASE TESTS
# =========================================================================

def test_synthetic_strong_semantic_noisy_ocr():
    """Edge Case 1: Strong semantic alignment, but heavily degraded OCR (noise)."""
    scorer = ConfidenceScorer()
    q_eval = QuestionEvaluation(
        question_id="Q_SYN_1",
        score=5.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="satisfied",
                marks_awarded=5.0,
                reason="Semantic match",
                semantic_similarity=0.72,
            )
        ],
        overall_reason="Satisfied",
    )
    # Severe OCR noise (0.35)
    res = scorer.evaluate_question_confidence(q_eval, ocr_confidence=0.35, student_text="noisy ocr text")
    assert res.confidence.label == "LOW"
    assert res.human_review_required is True
    assert "ocr" in res.confidence.reason.lower() or "handwriting" in res.confidence.reason.lower()


def test_synthetic_weak_semantic_evidence():
    """Edge Case 2: Weak semantic evidence requiring human review (LOW)."""
    scorer = ConfidenceScorer()
    q_eval = QuestionEvaluation(
        question_id="Q_SYN_2",
        score=1.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="not_satisfied",
                marks_awarded=1.0,
                reason="Vague and incomplete",
                semantic_similarity=0.22,
            )
        ],
        overall_reason="Incomplete",
    )
    res = scorer.evaluate_question_confidence(q_eval, ocr_confidence=0.60, student_text="some vague words")
    assert res.confidence.label == "LOW"
    assert res.human_review_required is True
    assert "manual review is required" in res.confidence.reason.lower()


def test_synthetic_explicit_contradiction():
    """Edge Case 3: Explicit contradiction produces HIGH confidence in negative decision."""
    scorer = ConfidenceScorer()
    q_eval = QuestionEvaluation(
        question_id="Q_SYN_3",
        score=0.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="contradicted",
                marks_awarded=0.0,
                reason="Direct contradiction detected",
                semantic_similarity=0.0,
            )
        ],
        overall_reason="Contradiction",
    )
    res = scorer.evaluate_question_confidence(q_eval, ocr_confidence=0.65, student_text="contradictory text")
    assert res.confidence.label == "HIGH"
    assert res.human_review_required is False
    assert "contradiction" in res.confidence.reason.lower()


def test_synthetic_ambiguous_evidence_near_boundary():
    """Edge Case 4: Evidence hovering near the decision threshold."""
    scorer = ConfidenceScorer()
    q_eval = QuestionEvaluation(
        question_id="Q_SYN_4",
        score=2.5,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="partially_satisfied",
                marks_awarded=2.5,
                reason="Borderline similarity",
                semantic_similarity=0.36,
            )
        ],
        overall_reason="Borderline",
    )
    res = scorer.evaluate_question_confidence(q_eval, ocr_confidence=0.60, student_text="this is a partial student answer text")
    assert res.confidence.label in ["MEDIUM", "LOW"]
    assert "threshold" in res.confidence.reason.lower() or "partial" in res.confidence.reason.lower()


def test_synthetic_confidence_reasons_generated_from_evidence():
    """Test D: Confidence reasons are generated from evaluation evidence rather than question-specific hardcoded strings.

    Demonstrates that:
    1. Full-credit reasons derive dynamically from the number of criteria.
    2. Contradiction reasons dynamically reference the specific criterion and explanation without hard-coding Q4.
    3. Weak evidence reasons recommend review based on semantic similarity.
    """
    scorer = ConfidenceScorer()

    # 1. Full credit case with arbitrary question ID
    q_eval_full = QuestionEvaluation(
        question_id="Q_ARBITRARY_10",
        score=5.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="satisfied",
                marks_awarded=2.5,
                reason="Concept matched",
                semantic_similarity=0.75,
            ),
            CriterionEvaluation(
                criterion_id="C2",
                status="satisfied",
                marks_awarded=2.5,
                reason="Example matched",
                semantic_similarity=0.80,
            ),
        ],
        overall_reason="All criteria satisfied",
    )
    res_full = scorer.evaluate_question_confidence(
        q_eval_full, ocr_confidence=0.75, student_text="A completely valid student answer with sufficient words"
    )
    assert res_full.confidence.label == "HIGH"
    # Reason must dynamically reflect the 2 criteria, not hardcoded question strings
    assert "across all 2 rubric criteria" in res_full.confidence.reason
    assert "Q1" not in res_full.confidence.reason
    assert "Q5" not in res_full.confidence.reason

    # 2. Contradiction case with arbitrary question ID and criterion
    q_eval_contra = QuestionEvaluation(
        question_id="Q_ARBITRARY_20",
        score=0.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="CUSTOM_CRIT_7",
                status="contradicted",
                marks_awarded=0.0,
                reason="Contradiction detected: asserted output instead of input",
                semantic_similarity=0.0,
            ),
        ],
        overall_reason="Contradiction detected",
    )
    res_contra = scorer.evaluate_question_confidence(
        q_eval_contra, ocr_confidence=0.70, student_text="Contradictory answer text"
    )
    assert res_contra.confidence.label == "HIGH"
    assert "CUSTOM_CRIT_7" in res_contra.confidence.reason
    assert "Q4" not in res_contra.confidence.reason

    # 3. Weak / Ambiguous case with arbitrary ID
    q_eval_weak = QuestionEvaluation(
        question_id="Q_ARBITRARY_30",
        score=1.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="not_satisfied",
                marks_awarded=1.0,
                reason="Weak semantic overlap",
                semantic_similarity=0.20,
            ),
        ],
        overall_reason="Incomplete answer",
    )
    res_weak = scorer.evaluate_question_confidence(
        q_eval_weak, ocr_confidence=0.50, student_text="vague few words"
    )
    assert res_weak.confidence.label == "LOW"
    assert res_weak.human_review_required is True
    assert "manual review is required" in res_weak.confidence.reason.lower() or "human review" in res_weak.confidence.reason.lower()


def test_synthetic_all_five_real_test_cases_expected_scores_and_confidences(confidence_data):
    """Test E: Verify that all five real test cases in student_001 produce expected results.

    Q1 = 5/5 (HIGH, review=False)
    Q2 = 5/5 (HIGH, review=False)
    Q3 = 5/5 (HIGH, review=False)
    Q4 = 0/5 (HIGH - contradiction, review=False)
    Q5 = 5/5 (LOW - handwriting uncertainty, review=True)
    Total = 20/25
    """
    expected = {
        "Q1": (5.0, "HIGH", False),
        "Q2": (5.0, "HIGH", False),
        "Q3": (5.0, "HIGH", False),
        "Q4": (0.0, "HIGH", False),
        "Q5": (5.0, "LOW", True),
    }
    total = 0.0
    for q in confidence_data.questions:
        exp_score, exp_conf, exp_rev = expected[q.question_id]
        assert q.score == exp_score
        assert q.confidence.label == exp_conf
        assert q.human_review_required is exp_rev
        total += q.score

    assert total == 20.0
    assert confidence_data.human_review_questions == ["Q5"]


def test_q5_low_confidence_and_human_review_due_to_handwriting_uncertainty(confidence_data):
    """Verify Q5 has full credit (5/5), LOW confidence, and requires human review due to noisy handwriting/OCR."""
    q5 = next(q for q in confidence_data.questions if q.question_id == "Q5")
    assert q5.score == 5.0
    assert q5.max_score == 5.0
    assert q5.confidence.label == "LOW"
    assert q5.human_review_required is True
    reason = q5.confidence.reason.lower()
    assert "ocr" in reason or "handwriting" in reason
    assert "human review is required" in reason or "uncertainty" in reason
    assert "Q5" in confidence_data.human_review_questions


def test_q4_high_confidence_despite_ocr_noise_due_to_strong_contradiction(confidence_data):
    """Verify Q4 retains HIGH confidence and 0/5 without human review because contradiction is strong."""
    q4 = next(q for q in confidence_data.questions if q.question_id == "Q4")
    assert q4.score == 0.0
    assert q4.confidence.label == "HIGH"
    assert q4.human_review_required is False
    assert "contradiction" in q4.confidence.reason.lower()


def test_synthetic_low_ocr_quality_produces_low_confidence_any_question_id():
    """Verify a synthetic answer with full semantic credit but high OCR uncertainty produces LOW confidence on any question ID."""
    scorer = ConfidenceScorer()
    q_eval = QuestionEvaluation(
        question_id="QUESTION_ARBITRARY_999",
        score=5.0,
        max_score=5.0,
        criterion_evaluations=[
            CriterionEvaluation(
                criterion_id="C1",
                status="satisfied",
                marks_awarded=5.0,
                reason="Semantically satisfied",
                semantic_similarity=0.48,
            )
        ],
        overall_reason="Satisfied",
    )
    # Severe OCR noise (< 0.45)
    res_severe = scorer.evaluate_question_confidence(
        q_eval, ocr_confidence=0.38, student_text="poorly recognized text"
    )
    assert res_severe.confidence.label == "LOW"
    assert res_severe.human_review_required is True
    assert "ocr" in res_severe.confidence.reason.lower() or "handwriting" in res_severe.confidence.reason.lower()

    # Elevated low-confidence token ratio (45% low-confidence tokens)
    res_high_ratio = scorer.evaluate_question_confidence(
        q_eval, ocr_confidence=0.61, student_text="partially recognized text", low_confidence_token_ratio=0.45
    )
    assert res_high_ratio.confidence.label == "LOW"
    assert res_high_ratio.human_review_required is True
    assert "45%" in res_high_ratio.confidence.reason


