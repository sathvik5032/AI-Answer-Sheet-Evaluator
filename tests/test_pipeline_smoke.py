"""Smoke test for the pipeline structure and placeholder classes."""

from src.config import Config
from src.models import (
    TextBoundingBox,
    OCRLine,
    OCRDocumentResult,
    ExtractedQuestionAnswer,
    RubricCriterion,
    RubricCriterionItem,
    QuestionRubric,
    MasterRubric,
    EvaluationResult,
    FinalEvaluationReport,
)
from src.preprocessing import DocumentPreprocessor
from src.ocr_engine import EasyOCREngine
from src.segmentation import QuestionSegmenter
from src.evaluator import SemanticEvaluator
from src.confidence import ConfidenceScorer
from src.reporter import ResultGenerator


def test_config_loading():
    """Verify that configuration loads without error."""
    config = Config("config/config.yaml")
    assert config.get("paths") is not None
    assert config.get("ocr") is not None


def test_models_instantiation():
    """Verify that core data models instantiate with valid schema."""
    bbox = TextBoundingBox(x_min=10, y_min=20, x_max=100, y_max=120)
    line = OCRLine(text="Question 1 Answer", confidence=0.88, bounding_box=bbox)
    doc_result = OCRDocumentResult(source_file="test.png", lines=[line])
    assert doc_result.source_file == "test.png"
    assert len(doc_result.lines) == 1

    rubric = RubricCriterion(
        question_id="Q1",
        question_text="What is RAM?",
        max_marks=5.0,
        model_answer="Random Access Memory is volatile primary memory.",
        key_points=["volatile", "primary memory", "read and write"]
    )
    assert rubric.max_marks == 5.0

    eval_result = EvaluationResult(
        question_id="Q1",
        max_marks=5.0,
        awarded_marks=4.5,
        similarity_score=0.89,
        confidence_level="HIGH",
        confidence_reason="Clear handwriting and strong key concept match."
    )
    assert eval_result.confidence_level == "HIGH"


def test_component_instantiation():
    """Verify all placeholder pipeline components instantiate cleanly."""
    preprocessor = DocumentPreprocessor()
    ocr = EasyOCREngine()
    segmenter = QuestionSegmenter()
    evaluator = SemanticEvaluator()
    scorer = ConfidenceScorer()
    reporter = ResultGenerator()

    assert preprocessor is not None
    assert ocr is not None
    assert segmenter is not None
    assert evaluator is not None
    assert scorer is not None
    assert reporter is not None


def test_rubric_schema_and_template():
    """Verify that computer_fundamentals.json loads and adheres to MasterRubric schema."""
    import json
    with open("data/rubrics/computer_fundamentals.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    master = MasterRubric(**data)
    assert master.subject == "Computer Fundamentals"
    assert len(master.questions) == 5
    for q in master.questions:
        assert q.question_id.startswith("Q")
        assert q.total_marks == 5.0
        assert len(q.prompt) > 0
        assert len(q.model_answer) > 0
        assert len(q.concepts) > 0
        assert len(q.criteria) > 0
        assert sum(c.marks for c in q.criteria) == q.total_marks

