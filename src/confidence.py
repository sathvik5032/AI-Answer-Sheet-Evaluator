"""Confidence Scoring module.

Assigns evaluation confidence ratings (HIGH / MEDIUM / LOW) and concise,
human-interpretable reasons for every evaluated question based on multi-factor
evidence (semantic alignment, criterion agreement, contradiction detection,
OCR noise, and decision boundary proximity).

Low-confidence cases are explicitly flagged for human review.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models import (
    ConfidenceDetail,
    QuestionConfidenceResult,
    QuestionEvaluation,
    SegmentedAnswer,
    SegmentedSubmissionResult,
    StudentConfidenceResult,
    StudentEvaluationResult,
)


class ConfidenceScorer:
    """Calculates multi-factor evaluation confidence levels and explanatory rationale."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.high_threshold = self.config.get("high_threshold", 0.70)
        self.medium_threshold = self.config.get("medium_threshold", 0.45)

    def evaluate_question_confidence(
        self,
        question_eval: QuestionEvaluation,
        ocr_confidence: float,
        student_text: str,
        ocr_items: Optional[List[Any]] = None,
        low_confidence_token_ratio: Optional[float] = None,
    ) -> QuestionConfidenceResult:
        """Determines the evaluation confidence for a single evaluated question.

        Multi-factor signals evaluated:
        1. Contradiction Evidence: An explicit factual contradiction provides strong,
           conclusive evidence for a 0-mark decision, resulting in HIGH confidence.
        2. Semantic Similarity Strength: Average and minimum similarity across criteria.
        3. Criterion Agreement: Consensus among independent criteria.
        4. OCR & Handwriting Quality: Low-confidence token proportion and mean OCR confidence.
           Answers with correct semantic scores but high handwriting/OCR uncertainty receive
           LOW confidence and are flagged for human review.
        5. Decision Boundary Proximity: Whether scores sit close to acceptance thresholds.

        Args:
            question_eval: The QuestionEvaluation result from Stage 6.
            ocr_confidence: Mean OCR extraction confidence [0.0 - 1.0].
            student_text: The segmented OCR answer text.
            ocr_items: Optional list of individual OCR items for granular noise analysis.
            low_confidence_token_ratio: Optional explicit ratio of low-confidence tokens (<0.50).

        Returns:
            QuestionConfidenceResult with label, reason, and human_review_required flag.
        """
        qid = question_eval.question_id
        score = question_eval.score
        max_score = question_eval.max_score
        c_evals = question_eval.criterion_evaluations

        # Check for explicit contradiction
        contradicted_criteria = [c for c in c_evals if c.status == "contradicted"]
        has_contradiction = len(contradicted_criteria) > 0

        sims = [c.semantic_similarity for c in c_evals if c.semantic_similarity > 0.0]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        min_sim = min(sims) if sims else 0.0

        clean_text = student_text.strip()
        word_count = len(clean_text.split())

        # Determine low-confidence token ratio (<0.50) from items if available
        if low_confidence_token_ratio is not None:
            low_conf_ratio = float(low_confidence_token_ratio)
        elif ocr_items:
            confs = []
            for item in ocr_items:
                if hasattr(item, "confidence"):
                    confs.append(float(item.confidence))
                elif isinstance(item, dict) and "confidence" in item:
                    confs.append(float(item["confidence"]))
            low_conf_ratio = (sum(1 for c in confs if c < 0.50) / len(confs)) if confs else 0.0
        else:
            low_conf_ratio = 0.0

        # --- Rule 1: Explicit Factual Contradiction ---
        if has_contradiction:
            label = "HIGH"
            human_review = False
            composite_score = 0.95
            first_contra = next((c for c in c_evals if c.status == "contradicted"), None)
            cid_str = f"in {first_contra.criterion_id} " if first_contra else ""
            contra_detail = f" ({first_contra.reason})" if (first_contra and first_contra.reason) else ""
            reason = (
                f"Explicit contradiction with the rubric requirement {cid_str}provides strong evidence for the 0-mark decision{contra_detail}."
            )

        # --- Rule 2: Blank or Empty Answer ---
        elif word_count == 0 or len(clean_text) < 3:
            label = "HIGH"
            human_review = False
            composite_score = 1.0
            reason = "Student answer is blank; 0 marks awarded with high confidence."

        # --- Rule 3: Full Credit (score == max_score) ---
        elif score == max_score:
            num_crit = len(c_evals)

            # Check if handwriting / OCR uncertainty undermines confidence in the full-credit evaluation:
            # Condition 1: Severe mean OCR degradation (< 0.45)
            # Condition 2: Substantial proportion of low-confidence OCR items (>= 40%)
            # Condition 3: Elevated low-confidence token ratio (>= 35%) combined with borderline semantic margin (min_sim < 0.45 or avg_sim < 0.50) or depressed OCR (< 0.62)
            is_ocr_uncertain = (
                ocr_confidence < 0.45
                or low_conf_ratio >= 0.40
                or (low_conf_ratio >= 0.35 and (min_sim < 0.45 or avg_sim < 0.50 or ocr_confidence < 0.62))
            )

            if is_ocr_uncertain:
                label = "LOW"
                human_review = True
                composite_score = min(0.40, 0.4 * avg_sim + 0.3 * ocr_confidence)
                pct_str = f" ({int(round(low_conf_ratio * 100))}% low-confidence tokens)" if low_conf_ratio > 0 else ""
                reason = (
                    f"Answer satisfies all {num_crit} rubric criteria, but handwriting and OCR quality "
                    f"introduce substantial uncertainty{pct_str}; human review is required."
                )
            elif avg_sim >= 0.50 and min_sim >= 0.45 and ocr_confidence >= 0.60 and low_conf_ratio < 0.35:
                label = "HIGH"
                human_review = False
                composite_score = min(1.0, 0.5 * avg_sim + 0.3 * ocr_confidence + 0.2)
                reason = f"Strong semantic alignment across all {num_crit} rubric criteria with consistent supporting evidence."
            elif ocr_confidence < 0.55 or min_sim < 0.40:
                label = "MEDIUM"
                human_review = False
                composite_score = 0.60
                reason = (
                    "Most criteria are supported, but OCR noise creates some ambiguity in conceptual phrasing."
                )
            else:
                label = "HIGH"
                human_review = False
                composite_score = 0.80
                reason = f"All {num_crit} rubric criteria are satisfied with consistent semantic evidence."

        # --- Rule 4: Partial Credit or Weak Evidence ---
        else:
            if avg_sim < 0.30 or word_count < 3 or ocr_confidence < 0.45:
                label = "LOW"
                human_review = True
                composite_score = 0.25
                reason = "Semantic evidence is weak and OCR extraction is ambiguous; manual review is required."
            elif 0.30 <= avg_sim <= 0.42:
                label = "MEDIUM"
                human_review = False
                composite_score = 0.55
                reason = (
                    "Partial semantic coverage near decision threshold; core concepts "
                    "partially match rubric criteria."
                )
            else:
                label = "MEDIUM"
                human_review = False
                composite_score = 0.65
                reason = "Partial marks awarded with moderate semantic certainty."

        confidence_detail = ConfidenceDetail(
            label=label,
            reason=reason,
            confidence_score=round(composite_score, 4),
            factors={
                "ocr_confidence": round(ocr_confidence, 4),
                "average_semantic_similarity": round(avg_sim, 4),
                "minimum_semantic_similarity": round(min_sim, 4),
                "word_count": word_count,
                "has_contradiction": has_contradiction,
                "low_confidence_token_ratio": round(low_conf_ratio, 4),
            },
        )

        return QuestionConfidenceResult(
            question_id=qid,
            score=score,
            max_score=max_score,
            confidence=confidence_detail,
            human_review_required=human_review,
        )

    def evaluate_submission_confidence(
        self,
        eval_result: StudentEvaluationResult,
        segmented_submission: SegmentedSubmissionResult,
    ) -> StudentConfidenceResult:
        """Evaluates confidence for all questions in a student submission."""
        seg_answers_map = {a.question_id: a for a in segmented_submission.answers}
        question_conf_results: List[QuestionConfidenceResult] = []
        human_review_list: List[str] = []

        for qe in eval_result.questions:
            qid = qe.question_id
            seg_ans = seg_answers_map.get(qid)
            ocr_conf = seg_ans.average_confidence if seg_ans else 0.0
            student_text = seg_ans.text if seg_ans else ""
            ocr_items = seg_ans.items if seg_ans else []

            q_conf = self.evaluate_question_confidence(
                question_eval=qe,
                ocr_confidence=ocr_conf,
                student_text=student_text,
                ocr_items=ocr_items,
            )
            question_conf_results.append(q_conf)

            if q_conf.human_review_required:
                human_review_list.append(qid)

        return StudentConfidenceResult(
            student_id=eval_result.student_id,
            questions=question_conf_results,
            human_review_questions=human_review_list,
            metadata={
                "total_questions": len(question_conf_results),
                "high_confidence_count": sum(1 for q in question_conf_results if q.confidence.label == "HIGH"),
                "medium_confidence_count": sum(1 for q in question_conf_results if q.confidence.label == "MEDIUM"),
                "low_confidence_count": sum(1 for q in question_conf_results if q.confidence.label == "LOW"),
                "human_review_required_count": len(human_review_list),
            },
        )

    def save_confidence_output(
        self,
        result: StudentConfidenceResult,
        output_path: str = "data/results/student_001_confidence.json",
    ) -> str:
        """Saves confidence scoring output to JSON."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        return str(out_file)

    def assess_confidence(
        self,
        ocr_confidence: float,
        semantic_similarity: float,
        answer_word_count: int,
        key_point_coverage: float,
    ) -> Tuple[str, str]:
        """Legacy helper for assessing confidence metrics."""
        composite = 0.4 * semantic_similarity + 0.3 * key_point_coverage + 0.3 * ocr_confidence
        if answer_word_count < 3:
            return "LOW", "Answer is too short or incomplete; manual review is required."
        if composite >= self.high_threshold:
            return "HIGH", "Strong semantic alignment across all rubric criteria with clear supporting evidence."
        elif composite >= self.medium_threshold:
            return "MEDIUM", "Most criteria are supported, but OCR noise creates some ambiguity in one required concept."
        else:
            return "LOW", "Semantic evidence is weak and OCR extraction is ambiguous; manual review is required."
