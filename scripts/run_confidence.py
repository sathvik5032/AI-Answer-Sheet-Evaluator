"""Runner script for Stage 7 Evaluation Confidence Scoring."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import ConfidenceScorer
from src.models import (
    SegmentedSubmissionResult,
    StudentConfidenceResult,
    StudentEvaluationResult,
)


def main():
    print("=" * 70)
    print("STAGE 7: RUNNING EVALUATION CONFIDENCE SCORING")
    print("=" * 70)

    eval_path = Path("data/results/student_001_evaluation.json")
    seg_path = Path("data/ocr_extracted/student_001_segmented.json")

    assert eval_path.is_file(), f"Evaluation result file not found: {eval_path}"
    assert seg_path.is_file(), f"Segmented OCR file not found: {seg_path}"

    # 1. Load Inputs
    print(f"\nLoading evaluation results: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_result = StudentEvaluationResult.model_validate(eval_data)

    print(f"Loading segmented answers: {seg_path}")
    with open(seg_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    segmented_submission = SegmentedSubmissionResult.model_validate(seg_data)

    # 2. Initialize Confidence Scorer
    scorer = ConfidenceScorer()

    # 3. Evaluate Confidence
    print("\nCalculating evaluation confidence across all questions...")
    confidence_result = scorer.evaluate_submission_confidence(
        eval_result=eval_result,
        segmented_submission=segmented_submission,
    )

    # 4. Save Structured Confidence Result
    output_path = Path("data/results/student_001_confidence.json")
    saved_path = scorer.save_confidence_output(confidence_result, str(output_path))
    print(f"\nSaved structured confidence result to: {saved_path}")

    # 5. Print Detailed Confidence Report
    print("\n" + "=" * 70)
    print("CONFIDENCE SCORING REPORT")
    print("=" * 70)
    print(f"Student ID:              {confidence_result.student_id}")
    print(f"Total Questions:         {len(confidence_result.questions)}")
    print(f"Human Review Required:   {len(confidence_result.human_review_questions)} questions ({confidence_result.human_review_questions})")

    for qc in confidence_result.questions:
        tag = f"[{qc.confidence.label}]"
        review_flag = " [HUMAN REVIEW REQUIRED]" if qc.human_review_required else ""
        print(f"\n--- {qc.question_id}: Score = {qc.score:.1f}/{qc.max_score:.1f} ---")
        print(f"  Confidence:            {tag}{review_flag}")
        print(f"  Reason:                \"{qc.confidence.reason}\"")
        factors = qc.confidence.factors
        print(f"  Underlying Signals:    OCR Conf = {factors.get('ocr_confidence', 0.0):.4f}, "
              f"Mean Sim = {factors.get('average_semantic_similarity', 0.0):.4f}, "
              f"Contradiction = {factors.get('has_contradiction', False)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
