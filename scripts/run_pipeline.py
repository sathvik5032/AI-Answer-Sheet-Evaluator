"""Master End-to-End Execution Pipeline for Answer Sheet Evaluation."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from scripts.run_ocr_pipeline import main as run_ocr
from scripts.run_segmentation import main as run_segmentation
from scripts.run_evaluation import main as run_evaluation
from scripts.run_confidence import main as run_confidence
from scripts.generate_final_reports import main as run_final_reports


def run_full_pipeline():
    print("=" * 80)
    print("STARTING END-TO-END ANSWER SHEET EVALUATION PIPELINE")
    print("Student ID: student_001 | Subject: Computer Fundamentals | Total Marks: 25.0")
    print("=" * 80)

    # 1. OCR Extraction (Stage 4)
    print("\n[Step 1/5] Executing OCR Extraction on answer sheets...")
    run_ocr()

    # 2. Question Segmentation (Stage 5)
    print("\n[Step 2/5] Executing Question Segmentation and Cross-Page Stitching...")
    run_segmentation()

    # 3. Semantic Evaluation (Stage 6)
    print("\n[Step 3/5] Executing Semantic Rubric Evaluation...")
    run_evaluation()

    # 4. Confidence Scoring (Stage 7)
    print("\n[Step 4/5] Executing Multi-Factor Confidence Scoring...")
    run_confidence()

    # 5. Final Reporting (Stage 8)
    print("\n[Step 5/5] Generating Final Standardized JSON & CSV Reports...")
    run_final_reports()

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("Final Deliverables:")
    print("  - JSON: data/final/student_001_final_evaluation.json")
    print("  - CSV:  data/final/student_001_final_evaluation.csv")
    print("  - Note: data/final/approach_note.md")
    print("  - Docs: data/final/README.md")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
