"""Main pipeline orchestrator and CLI entry point for Answer Sheet Evaluation."""

import argparse
import sys
from pathlib import Path
from src.config import Config
from src.preprocessing import DocumentPreprocessor
from src.ocr_engine import EasyOCREngine
from src.segmentation import QuestionSegmenter
from src.evaluator import SemanticEvaluator
from src.confidence import ConfidenceScorer
from src.reporter import ResultGenerator


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated AI/ML Evaluation of Handwritten Answer Sheets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--answer-sheet",
        type=str,
        default=None,
        help="Path to input handwritten answer sheet image or PDF",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Path to answer key / rubric JSON or YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Directory to save evaluation reports",
    )
    return parser.parse_args()


def main() -> int:
    """Initializes and coordinates the evaluation pipeline."""
    args = parse_arguments()
    print("=" * 65)
    print("AI/ML Engineer - Answer Sheet Evaluation Pipeline (Skeleton)")
    print("Subject: Computer Fundamentals")
    print("=" * 65)

    # 1. Load configuration
    config = Config(args.config)
    print(f"[+] Loaded configuration from: {args.config}")

    # 2. Instantiate pipeline stage components (placeholders)
    preprocessor = DocumentPreprocessor(config.get("preprocessing"))
    ocr_engine = EasyOCREngine(
        languages=config.get("ocr", {}).get("languages", ["en"]),
        use_gpu=config.get("ocr", {}).get("use_gpu", False)
    )
    segmenter = QuestionSegmenter(
        delimiter_patterns=config.get("segmentation", {}).get("question_delimiters", [])
    )
    evaluator = SemanticEvaluator(
        model_name=config.get("evaluation", {}).get("model_name", "all-MiniLM-L6-v2"),
        config=config.get("evaluation")
    )
    confidence_scorer = ConfidenceScorer(
        thresholds=config.get("confidence", {}).get("thresholds")
    )
    reporter = ResultGenerator(
        output_dir=args.output_dir
    )

    print("[+] Pipeline components instantiated successfully:")
    print(f"    - Preprocessor       : {preprocessor.__class__.__name__}")
    print(f"    - OCR Engine         : {ocr_engine.__class__.__name__}")
    print(f"    - Segmenter          : {segmenter.__class__.__name__}")
    print(f"    - Semantic Evaluator : {evaluator.__class__.__name__}")
    print(f"    - Confidence Scorer  : {confidence_scorer.__class__.__name__}")
    print(f"    - Result Generator   : {reporter.__class__.__name__}")
    print("-" * 65)
    print("[*] Stage 1 skeleton initialized. Pipeline ready for stage-by-stage implementation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
