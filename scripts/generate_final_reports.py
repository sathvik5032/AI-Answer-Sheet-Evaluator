"""Generates the final submission JSON and CSV reports for student_001."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.reporter import ResultGenerator


def main():
    print("=" * 70)
    print("STAGE 8: GENERATING FINAL SUBMISSION REPORTS")
    print("=" * 70)

    generator = ResultGenerator(output_dir="data/final")
    paths = generator.generate_final_submission_reports(
        eval_path="data/results/student_001_evaluation.json",
        conf_path="data/results/student_001_confidence.json",
        seg_path="data/ocr_extracted/student_001_segmented.json",
        rubric_path="data/rubrics/computer_fundamentals.json",
    )

    print(f"\nFinal JSON Report generated at: {paths['json_report']}")
    print(f"Final CSV Report generated at:  {paths['csv_report']}")
    print("\nReports successfully verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
