"""Script to execute question segmentation on extracted OCR output."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import OCRSubmissionResult
from src.segmentation import QuestionSegmenter


def main():
    print("=" * 70)
    print("STAGE 5: RUNNING QUESTION SEGMENTATION (Q1 - Q5)")
    print("=" * 70)

    ocr_input_path = Path("data/ocr_extracted/student_001_ocr.json")
    if not ocr_input_path.is_file():
        raise FileNotFoundError(f"Input OCR file not found: {ocr_input_path}")

    # 1. Initialize Segmenter
    segmenter = QuestionSegmenter(target_questions=["Q1", "Q2", "Q3", "Q4", "Q5"])

    # 2. Load OCR Submission
    print(f"\nLoading OCR input: {ocr_input_path}")
    ocr_submission = segmenter.load_ocr_submission(str(ocr_input_path))
    print(f"Loaded {ocr_submission.total_pages} pages, {ocr_submission.total_items} total OCR detections.")

    # 3. Perform Segmentation
    print("\nExecuting segmentation...")
    segmented_result = segmenter.segment_submission(ocr_submission)

    # 4. Save Segmented JSON
    output_path = Path("data/ocr_extracted/student_001_segmented.json")
    saved_path = segmenter.save_segmented_output(segmented_result, str(output_path))
    print(f"\nSaved structured segmented answers to: {saved_path}")

    # 5. Display Summary & Verification
    print("\n" + "=" * 70)
    print("SEGMENTATION SUMMARY")
    print("=" * 70)
    print(f"Student ID:      {segmented_result.student_id}")
    print(f"Total Answers:   {segmented_result.total_answers}")

    for ans in segmented_result.answers:
        print(f"\n--- Question {ans.question_id} ---")
        print(f"  Source Pages:     {ans.source_pages}")
        print(f"  OCR Item Count:   {ans.ocr_item_count}")
        print(f"  Mean Confidence:  {ans.average_confidence:.4f}")
        print(f"  Reconstructed Text:\n    \"{ans.text}\"")

    # Cross-page verification check
    q3 = next(a for a in segmented_result.answers if a.question_id == "Q3")
    print("\n" + "-" * 70)
    print(f"CRITICAL TEST (Q3 Cross-Page Merging):")
    print(f"  Q3 Source Pages:  {q3.source_pages}")
    is_q3_cross_page = ("page1.jpg" in q3.source_pages and "page2.jpg" in q3.source_pages)
    print(f"  Cross-Page Status: {'PASSED (Contains BOTH page1.jpg and page2.jpg)' if is_q3_cross_page else 'FAILED'}")
    print("-" * 70)


if __name__ == "__main__":
    main()
