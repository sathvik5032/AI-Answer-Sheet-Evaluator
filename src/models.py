"""Data models and schemas for the Answer Sheet Evaluation pipeline."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TextBoundingBox(BaseModel):
    """Bounding box coordinates for detected text tokens/lines."""
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0


class OCRLine(BaseModel):
    """Represents a single recognized line of text with its confidence score."""
    text: str
    confidence: float = 0.0
    bounding_box: Optional[TextBoundingBox] = None


class OCRDocumentResult(BaseModel):
    """Raw result returned by the OCR/HTR engine for an entire document or page."""
    source_file: str
    full_text: str = ""
    lines: List[OCRLine] = Field(default_factory=list)
    average_confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OCRItem(BaseModel):
    """Represents an individual detected text block/line with confidence and bbox."""
    order: int
    page_number: int
    text: str
    confidence: float
    bbox: List[List[int]] = Field(default_factory=list)
    bounding_box: Optional[TextBoundingBox] = None


class OCRPageResult(BaseModel):
    """Structured OCR results for a single page of an answer sheet."""
    page_number: int
    source_file: str
    image_size: List[int] = Field(default_factory=list)  # [width, height]
    items_count: int = 0
    average_confidence: float = 0.0
    page_text: str = ""
    items: List[OCRItem] = Field(default_factory=list)


class OCRSubmissionResult(BaseModel):
    """Complete multi-page OCR extraction result for a student submission."""
    student_id: str = "student_001"
    total_pages: int = 0
    total_items: int = 0
    overall_confidence: float = 0.0
    pages: List[OCRPageResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedQuestionAnswer(BaseModel):
    """Segmented answer belonging to a specific question number."""
    question_id: str
    raw_text: str
    ocr_confidence: float = 0.0
    source_lines: List[OCRLine] = Field(default_factory=list)


class SegmentedAnswerItem(BaseModel):
    """Reference to an individual OCR detection item within a segmented answer."""
    page_number: int
    source_file: str
    order: int
    text: str
    confidence: float
    bbox: List[List[int]] = Field(default_factory=list)
    bounding_box: Optional[TextBoundingBox] = None


class SegmentedAnswer(BaseModel):
    """A segmented student answer corresponding to a specific question ID."""
    question_id: str
    source_pages: List[str] = Field(default_factory=list)
    text: str = ""
    ocr_item_count: int = 0
    average_confidence: float = 0.0
    items: List[SegmentedAnswerItem] = Field(default_factory=list)


class SegmentedSubmissionResult(BaseModel):
    """Overall segmented result containing all extracted question answers for a submission."""
    student_id: str = "student_001"
    total_answers: int = 5
    answers: List[SegmentedAnswer] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)



class RubricCriterionItem(BaseModel):
    """Specific scoring criterion with dedicated point allocation."""
    criterion_id: Optional[str] = None
    description: Optional[str] = None
    criterion: Optional[str] = None
    marks: float

    @property
    def text(self) -> str:
        return self.description or self.criterion or ""


class RubricCriterion(BaseModel):
    """Grading rubric and reference answer for a single question (legacy/compact format)."""
    question_id: str
    question_text: str
    max_marks: float
    model_answer: str
    key_points: List[str] = Field(default_factory=list)
    rubric_criteria: List[RubricCriterionItem] = Field(default_factory=list)


class QuestionRubric(BaseModel):
    """Standardized question-level rubric supporting key concepts and criterion breakdowns."""
    question_id: str
    question_text: Optional[str] = None
    question: Optional[str] = None
    maximum_marks: Optional[float] = None
    max_marks: Optional[float] = None
    expected_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    key_concepts: List[str] = Field(default_factory=list)
    expected_concepts: List[str] = Field(default_factory=list)
    rubric_criteria: List[RubricCriterionItem] = Field(default_factory=list)
    rubric: List[RubricCriterionItem] = Field(default_factory=list)

    @property
    def prompt(self) -> str:
        return self.question or self.question_text or ""

    @property
    def total_marks(self) -> float:
        return self.max_marks if self.max_marks is not None else (self.maximum_marks or 0.0)

    @property
    def model_answer(self) -> str:
        return self.reference_answer or self.expected_answer or ""

    @property
    def concepts(self) -> List[str]:
        return self.expected_concepts if self.expected_concepts else self.key_concepts

    @property
    def criteria(self) -> List[RubricCriterionItem]:
        return self.rubric if self.rubric else self.rubric_criteria


class MasterRubric(BaseModel):
    """Master subject-level rubric containing a collection of question rubrics."""
    subject: str = "Computer Fundamentals"
    total_marks: float = 0.0
    description: Optional[str] = ""
    version: Optional[str] = "1.0"
    questions: List[QuestionRubric] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Evaluation score and confidence assessment for a single question answer."""
    question_id: str
    max_marks: float
    awarded_marks: float
    similarity_score: float
    confidence_level: str  # e.g., "HIGH", "MEDIUM", "LOW"
    confidence_reason: str
    feedback: str = ""


class FinalEvaluationReport(BaseModel):
    """Comprehensive evaluation report for an answer sheet."""
    student_id: Optional[str] = "Student_01"
    answer_sheet_file: str
    timestamp: str
    total_max_marks: float
    total_awarded_marks: float
    percentage: float
    results: List[EvaluationResult] = Field(default_factory=list)


class CriterionEvaluation(BaseModel):
    """Evaluation result for a single rubric criterion."""
    criterion_id: str
    status: str  # "satisfied", "partially_satisfied", "not_satisfied", "contradicted"
    marks_awarded: float
    reason: str
    semantic_similarity: float = 0.0


class QuestionEvaluation(BaseModel):
    """Semantic evaluation result for a single question."""
    question_id: str
    score: float
    max_score: float
    criterion_evaluations: List[CriterionEvaluation] = Field(default_factory=list)
    overall_reason: str


class StudentEvaluationResult(BaseModel):
    """Structured semantic evaluation output for a complete student submission."""
    student_id: str = "student_001"
    subject: str = "Computer Fundamentals"
    questions: List[QuestionEvaluation] = Field(default_factory=list)
    total_score: float = 0.0
    total_max_score: float = 25.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceDetail(BaseModel):
    """Confidence label and one-line explanatory justification."""
    label: str  # "HIGH", "MEDIUM", "LOW"
    reason: str
    confidence_score: float = 0.0
    factors: Dict[str, Any] = Field(default_factory=dict)


class QuestionConfidenceResult(BaseModel):
    """Question-level evaluation confidence assessment."""
    question_id: str
    score: float
    max_score: float
    confidence: ConfidenceDetail
    human_review_required: bool = False


class StudentConfidenceResult(BaseModel):
    """Overall confidence evaluation report for a student submission."""
    student_id: str = "student_001"
    questions: List[QuestionConfidenceResult] = Field(default_factory=list)
    human_review_questions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


