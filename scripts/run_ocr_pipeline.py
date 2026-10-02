"""Script to execute handwriting OCR extraction across page1.jpg and page2.jpg."""

import json
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.config import Config
from src.models import OCRSubmissionResult
from src.ocr_engine import EasyOCREngine
from src.preprocessing import DocumentPreprocessor


def main():
    print("=" * 70)
    print("STAGE 4: RUNNING HANDWRITTEN ANSWER SHEET OCR EXTRACTION")
    print("=" * 70)

    # 1. Load Configuration
    config = Config("config/config.yaml")
    ocr_cfg = config.get("ocr", {})
    prep_cfg = config.get("preprocessing", {})

    print(f"OCR Engine configured: {ocr_cfg.get('engine', 'easyocr')}")
    print(f"Languages: {ocr_cfg.get('languages', ['en'])}")
    print(f"Use GPU: {ocr_cfg.get('use_gpu', False)}")

    # 2. Initialize Preprocessor and Engine
    preprocessor = DocumentPreprocessor(config=prep_cfg)
    ocr_engine = EasyOCREngine(
        languages=ocr_cfg.get("languages", ["en"]),
        use_gpu=ocr_cfg.get("use_gpu", False),
        min_confidence=ocr_cfg.get("min_confidence", 0.0),
    )

    # Pre-initialize reader
    print("\nInitializing EasyOCR reader model...")
    t_init_start = time.time()
    ocr_engine.initialize_reader()
    print(f"Reader initialized in {time.time() - t_init_start:.2f}s")

    # 3. Define target pages
    answer_sheet_files = [
        ("data/inputs/answer_sheets/page1.jpg", "page1.jpg", 1),
        ("data/inputs/answer_sheets/page2.jpg", "page2.jpg", 2),
    ]

    preprocessed_pages = []
    prep_stats = []

    print("\n--- Preprocessing Pages ---")
    for file_path, source_name, page_num in answer_sheet_files:
        t0 = time.time()
        raw_images = preprocessor.load_document(file_path)
        assert len(raw_images) == 1, f"Expected 1 image for {file_path}"
        raw_img = raw_images[0]

        # Apply preprocessing
        proc_img, meta = preprocessor.preprocess_image(raw_img)
        prep_time = time.time() - t0

        preprocessed_pages.append((proc_img, source_name, page_num))
        prep_stats.append({
            "page": page_num,
            "file": source_name,
            "raw_shape": meta["original_shape"],
            "deskew_angle": meta["deskew_angle"],
            "contrast_enhanced": meta["contrast_enhanced"],
            "prep_time_sec": round(prep_time, 3),
        })
        print(f"Page {page_num} ({source_name}): Preprocessed in {prep_time:.3f}s | Deskew: {meta['deskew_angle']:.2f} deg | Shape: {meta['final_shape']}")

    # 4. Execute OCR extraction per page
    print("\n--- Running EasyOCR Extraction ---")
    page_results = []
    ocr_timings = []

    for img, source_name, page_num in preprocessed_pages:
        print(f"\nProcessing Page {page_num} ({source_name})...")
        t_page = time.time()
        page_res = ocr_engine.extract_page(img, page_number=page_num, source_file=source_name)
        elapsed = time.time() - t_page
        ocr_timings.append(elapsed)
        page_results.append(page_res)

        print(f"  Done in {elapsed:.2f}s")
        print(f"  Detections: {page_res.items_count} items")
        print(f"  Average Confidence: {page_res.average_confidence:.4f}")
        confs = [it.confidence for it in page_res.items]
        if confs:
            print(f"  Min Confidence: {min(confs):.4f} | Max Confidence: {max(confs):.4f}")

    # 5. Aggregate into single OCRSubmissionResult
    total_items = sum(p.items_count for p in page_results)
    all_confs = [it.confidence for p in page_results for it in p.items]
    overall_conf = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0

    submission_result = OCRSubmissionResult(
        student_id="student_001",
        total_pages=len(page_results),
        total_items=total_items,
        overall_confidence=overall_conf,
        pages=page_results,
        metadata={
            "ocr_engine": "EasyOCR",
            "ocr_version": "1.7.2",
            "preprocessing": prep_stats,
            "ocr_timing_seconds": [round(t, 2) for t in ocr_timings],
            "total_inference_time_seconds": round(sum(ocr_timings), 2),
            "cross_page_note": "Q3 begins on Page 1 and concludes on Page 2.",
        },
    )

    # 6. Save structured output
    output_dir = Path("data/ocr_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "student_001_ocr.json"

    saved_path = ocr_engine.save_submission_output(submission_result, str(output_file))
    print(f"\nSaved structured OCR output to: {saved_path}")

    # 7. Print summary & sample extractions
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total Pages Processed: {submission_result.total_pages}")
    print(f"Total Detected Items:  {submission_result.total_items}")
    print(f"Overall Mean Conf:     {submission_result.overall_confidence:.4f}")
    print(f"Total Inference Time:  {sum(ocr_timings):.2f}s")

    for p in submission_result.pages:
        print(f"\n--- Page {p.page_number} ({p.source_file}) Sample (first 5 items) ---")
        for item in p.items[:5]:
            print(f"  [{item.order:2d}] (conf: {item.confidence:0.4f}, bbox: {item.bounding_box.x_min},{item.bounding_box.y_min}) -> \"{item.text}\"")

        print(f"\n--- Page {p.page_number} ({p.source_file}) Sample (last 3 items) ---")
        for item in p.items[-3:]:
            print(f"  [{item.order:2d}] (conf: {item.confidence:0.4f}, bbox: {item.bounding_box.x_min},{item.bounding_box.y_min}) -> \"{item.text}\"")


if __name__ == "__main__":
    main()
