"""OCR / Handwritten Text Recognition (HTR) module.

Extracts text, bounding boxes, reading order, and line-level confidence scores
from handwritten answer sheet pages using pre-trained EasyOCR models.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import (
    OCRDocumentResult,
    OCRItem,
    OCRLine,
    OCRPageResult,
    OCRSubmissionResult,
    TextBoundingBox,
)


class BaseOCREngine(ABC):
    """Abstract base class for OCR/HTR recognition engines."""

    @abstractmethod
    def extract_text(self, preprocessed_images: List[Any], source_file: str) -> OCRDocumentResult:
        """Extracts text and confidences from page images."""
        pass


class EasyOCREngine(BaseOCREngine):
    """Pre-trained EasyOCR engine wrapper for handwritten text recognition."""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        min_confidence: float = 0.0,
    ):
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.reader = None

    def initialize_reader(self) -> None:
        """Initializes the underlying pre-trained EasyOCR reader model."""
        if self.reader is None:
            # Set OpenMP and encoding compatibility flags for Windows
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["PYTHONIOENCODING"] = "utf-8"
            import sys
            if hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            if hasattr(sys.stderr, "reconfigure"):
                try:
                    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)

    def extract_page(
        self,
        image: np.ndarray,
        page_number: int,
        source_file: str,
    ) -> OCRPageResult:
        """Runs OCR on a single page image and structures detected items.

        Args:
            image: Preprocessed page image array (BGR or RGB).
            page_number: Sequential 1-based page index (e.g., 1 or 2).
            source_file: Filename of the source image.

        Returns:
            OCRPageResult containing ordered items, coordinates, text, and confidences.
        """
        self.initialize_reader()

        raw_results = self.reader.readtext(image)

        # Structure each detection into raw items
        parsed_items: List[Dict[str, Any]] = []
        for bbox, text, prob in raw_results:
            conf = float(prob)
            if conf < self.min_confidence:
                continue

            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue

            # Convert bbox points to integers [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
            xs = [pt[0] for pt in int_bbox]
            ys = [pt[1] for pt in int_bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            parsed_items.append({
                "text": cleaned_text,
                "confidence": conf,
                "bbox": int_bbox,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            })

        # Sort items in natural top-to-bottom, left-to-right reading order
        # Approximate vertical line clustering using line tolerance
        line_band_height = 20  # Pixels band for grouping horizontal lines
        parsed_items.sort(key=lambda it: (it["y_min"] // line_band_height, it["x_min"]))

        # Build typed OCRItem models
        ocr_items: List[OCRItem] = []
        for idx, item in enumerate(parsed_items):
            box = TextBoundingBox(
                x_min=item["x_min"],
                y_min=item["y_min"],
                x_max=item["x_max"],
                y_max=item["y_max"],
            )
            ocr_item = OCRItem(
                order=idx + 1,
                page_number=page_number,
                text=item["text"],
                confidence=round(item["confidence"], 4),
                bbox=item["bbox"],
                bounding_box=box,
            )
            ocr_items.append(ocr_item)

        h, w = image.shape[:2]
        avg_conf = (
            round(sum(it.confidence for it in ocr_items) / len(ocr_items), 4)
            if ocr_items
            else 0.0
        )
        page_text = "\n".join(it.text for it in ocr_items)

        return OCRPageResult(
            page_number=page_number,
            source_file=source_file,
            image_size=[w, h],
            items_count=len(ocr_items),
            average_confidence=avg_conf,
            page_text=page_text,
            items=ocr_items,
        )

    def extract_submission(
        self,
        pages_data: List[Tuple[np.ndarray, str, int]],
        student_id: str = "student_001",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OCRSubmissionResult:
        """Processes all pages of a student submission while preserving page separation.

        Args:
            pages_data: List of tuples (image_array, source_filename, page_number).
            student_id: Student submission identifier.
            metadata: Optional execution metadata.

        Returns:
            OCRSubmissionResult aggregating all pages under one student submission.
        """
        self.initialize_reader()
        page_results: List[OCRPageResult] = []

        for image, source_file, page_num in pages_data:
            page_res = self.extract_page(image, page_number=page_num, source_file=source_file)
            page_results.append(page_res)

        total_items = sum(p.items_count for p in page_results)
        all_confidences = [
            item.confidence
            for p in page_results
            for item in p.items
        ]
        overall_conf = (
            round(sum(all_confidences) / len(all_confidences), 4)
            if all_confidences
            else 0.0
        )

        return OCRSubmissionResult(
            student_id=student_id,
            total_pages=len(page_results),
            total_items=total_items,
            overall_confidence=overall_conf,
            pages=page_results,
            metadata=metadata or {},
        )

    def extract_text(self, preprocessed_images: List[Any], source_file: str) -> OCRDocumentResult:
        """Extracts text and confidences from images (implements BaseOCREngine).

        Args:
            preprocessed_images: List of image arrays.
            source_file: Name of the source file.

        Returns:
            OCRDocumentResult containing legacy OCRLine models.
        """
        self.initialize_reader()
        all_lines: List[OCRLine] = []

        for page_idx, img in enumerate(preprocessed_images):
            page_res = self.extract_page(img, page_number=page_idx + 1, source_file=source_file)
            for item in page_res.items:
                all_lines.append(
                    OCRLine(
                        text=item.text,
                        confidence=item.confidence,
                        bounding_box=item.bounding_box,
                    )
                )

        full_text = "\n".join(l.text for l in all_lines)
        avg_conf = (
            round(sum(l.confidence for l in all_lines) / len(all_lines), 4)
            if all_lines
            else 0.0
        )
        return OCRDocumentResult(
            source_file=source_file,
            full_text=full_text,
            lines=all_lines,
            average_confidence=avg_conf,
        )

    def save_submission_output(
        self,
        result: OCRSubmissionResult,
        output_path: str,
    ) -> str:
        """Saves structured OCR submission results to a JSON file.

        Args:
            result: The OCRSubmissionResult instance.
            output_path: Path where the JSON file will be written.

        Returns:
            Path of the saved JSON file.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        return str(out_path)
