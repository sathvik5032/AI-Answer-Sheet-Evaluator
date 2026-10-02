"""Runner script for Stage 6 Semantic Answer Evaluation."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import SemanticEvaluator
from src.models import MasterRubric, SegmentedSubmissionResult


def main():
    print("=" * 70)
    print("STAGE 6: RUNNING SEMANTIC ANSWER EVALUATION")
    print("=" * 70)

    # 1. Input paths
    segmented_path = Path("data/ocr_extracted/student_001_segmented.json")
    rubric_path = Path("data/rubrics/computer_fundamentals.json")
    qp_path = Path("data/inputs/question_paper/Computer_Fundamentals_Question_Paper.pdf")
    gt_path = Path("data/references/ground_truth.json")

    assert segmented_path.is_file(), f"Segmented OCR file not found: {segmented_path}"
    assert rubric_path.is_file(), f"Rubric file not found: {rubric_path}"
    assert qp_path.is_file(), f"Question paper file not found: {qp_path}"

    # 2. Load inputs
    print(f"\nLoading segmented student answers: {segmented_path}")
    with open(segmented_path, "r", encoding="utf-8") as f:
        segmented_data = json.load(f)
    segmented_submission = SegmentedSubmissionResult.model_validate(segmented_data)

    print(f"Loading rubric: {rubric_path}")
    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric_data = json.load(f)
    master_rubric = MasterRubric.model_validate(rubric_data)

    # 3. Initialize Evaluator
    evaluator = SemanticEvaluator(model_name="all-MiniLM-L6-v2")
    print(f"Semantic Evaluator initialized with model: {evaluator.model_name}")

    # 4. Run Evaluation
    print("\nExecuting semantic evaluation across all questions...")
    evaluation_result = evaluator.evaluate_submission(segmented_submission, master_rubric)

    # 5. Save structured evaluation result
    output_path = Path("data/results/student_001_evaluation.json")
    saved_path = evaluator.save_evaluation_result(evaluation_result, str(output_path))
    print(f"\nSaved structured evaluation result to: {saved_path}")

    # 6. Print detailed per-question report
    print("\n" + "=" * 70)
    print("SEMANTIC EVALUATION BREAKDOWN")
    print("=" * 70)
    print(f"Student ID:    {evaluation_result.student_id}")
    print(f"Subject:       {evaluation_result.subject}")
    print(f"Total Score:   {evaluation_result.total_score:.1f} / {evaluation_result.total_max_score:.1f}")

    for qe in evaluation_result.questions:
        print(f"\n{'='*20} {qe.question_id} (Score: {qe.score:.1f} / {qe.max_score:.1f}) {'='*20}")
        print(f"Overall Reason: {qe.overall_reason}")
        print("Criterion-Level Evaluations:")
        for ce in qe.criterion_evaluations:
            status_tag = f"[{ce.status.upper()}]"
            print(f"  - {ce.criterion_id} {status_tag:15s} Awarded: {ce.marks_awarded:.1f}m | {ce.reason}")

    # 7. Validation against ground truth (for diagnostic reporting only)
    if gt_path.is_file():
        print("\n" + "=" * 70)
        print("VALIDATION AGAINST GROUND TRUTH (data/references/ground_truth.json)")
        print("=" * 70)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_answers_map = {a["question_id"]: a for a in gt_data.get("answers", [])}
        expected_total = gt_data.get("total_expected_marks", 20)

        match_count = 0
        for qe in evaluation_result.questions:
            gt_entry = gt_answers_map.get(qe.question_id, {})
            expected_m = gt_entry.get("expected_marks")
            case_name = gt_entry.get("evaluation_case", "unknown")
            is_match = (qe.score == expected_m)
            if is_match:
                match_count += 1
            status_str = "MATCH" if is_match else f"MISMATCH (Expected: {expected_m})"
            print(f"  {qe.question_id}: Awarded = {qe.score:.1f}, Expected = {expected_m} ({case_name}) -> {status_str}")

        print(f"\nTotal Awarded Marks:  {evaluation_result.total_score:.1f}")
        print(f"Total Expected Marks: {expected_total}")
        print(f"Ground Truth Status:  {'PERFECT ALIGNMENT (5/5 questions matched)' if match_count == 5 else f'{match_count}/5 questions matched'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
