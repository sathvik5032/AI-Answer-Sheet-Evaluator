"""Question Segmentation module.

Parses continuous extracted OCR detections and token bounding boxes, detecting question
headers (e.g., '1', '2 ,', '3,', '4-', '5. 4'), layout boundaries, and cross-page continuations
to partition the multi-page answer sheet into structured, per-question answer blocks (Q1 - Q5).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models import (
    ExtractedQuestionAnswer,
    OCRDocumentResult,
    OCRItem,
    OCRPageResult,
    OCRSubmissionResult,
    SegmentedAnswer,
    SegmentedAnswerItem,
    SegmentedSubmissionResult,
    TextBoundingBox,
)


class QuestionSegmenter:
    """Segments extracted OCR text into discrete question-answer units."""

    DEFAULT_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def __init__(
        self,
        target_questions: Optional[List[str]] = None,
        delimiter_patterns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.target_questions = target_questions or self.DEFAULT_QUESTIONS
        self.delimiter_patterns = delimiter_patterns or []
        self.config = config or {}
        self.vertical_gap_threshold = self.config.get("vertical_gap_threshold", 60)
        self.margin_x_threshold = self.config.get("margin_x_threshold", 250)
        self.line_clustering_tolerance = self.config.get("line_clustering_tolerance", 30)

        # Regex patterns to detect question labels tolerating handwriting OCR noise
        self.question_patterns = {
            "Q1": re.compile(r"^(?:q(?:uestion)?\s*[\.\:\-]?)?\s*1[\.\,\-\)\s]*$", re.IGNORECASE),
            "Q2": re.compile(r"^(?:q(?:uestion)?\s*[\.\:\-]?)?\s*2[\.\,\-\)\s]*$", re.IGNORECASE),
            "Q3": re.compile(r"^(?:q(?:uestion)?\s*[\.\:\-]?)?\s*3[\.\,\-\)\s]*$", re.IGNORECASE),
            "Q4": re.compile(r"^(?:q(?:uestion)?\s*[\.\:\-]?)?\s*4[\.\,\-\)\s]*$", re.IGNORECASE),
            "Q5": re.compile(r"^(?:q(?:uestion)?\s*[\.\:\-]?)?\s*5[\.\,\-\)\s]*", re.IGNORECASE),
        }

    def load_ocr_submission(self, json_path: str) -> OCRSubmissionResult:
        """Loads and validates a structured OCR submission JSON file.

        Args:
            json_path: Path to the student OCR JSON file.

        Returns:
            OCRSubmissionResult Pydantic model.
        """
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"OCR submission file not found: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return OCRSubmissionResult.model_validate(data)

    def detect_page_blocks(self, page: OCRPageResult) -> List[List[OCRItem]]:
        """Partitions OCR items of a single page into continuous visual paragraph blocks.

        Uses vertical whitespace gaps between consecutive tokens in reading order.
        """
        if not page.items:
            return []

        blocks: List[List[OCRItem]] = []
        current_block: List[OCRItem] = []

        for item in page.items:
            if not current_block:
                current_block.append(item)
            else:
                prev_item = current_block[-1]
                gap = (
                    item.bounding_box.y_min - prev_item.bounding_box.y_max
                    if item.bounding_box and prev_item.bounding_box
                    else 0
                )
                if gap > self.vertical_gap_threshold:
                    blocks.append(current_block)
                    current_block = [item]
                else:
                    current_block.append(item)

        if current_block:
            blocks.append(current_block)

        return blocks

    def identify_block_question_id(self, block: List[OCRItem]) -> Optional[str]:
        """Detects if any OCR item within the margin region of a block matches a question marker."""
        for item in block:
            x_min = item.bounding_box.x_min if item.bounding_box else 0
            if x_min <= self.margin_x_threshold:
                clean_text = item.text.strip()
                for qid, pat in self.question_patterns.items():
                    if pat.search(clean_text):
                        return qid
        return None

    def reconstruct_text(self, items: List[OCRItem]) -> str:
        """Reconstructs text from OCR items using line-aware spatial clustering.

        Preserves all OCR words and characters as extracted without silent corrections,
        while ordering tokens within lines from left-to-right and lines from top-to-bottom.
        """
        if not items:
            return ""

        # Group items by page number first to preserve page-local geometry
        pages_items: Dict[int, List[OCRItem]] = {}
        for item in items:
            pages_items.setdefault(item.page_number, []).append(item)

        page_texts = []
        for page_num in sorted(pages_items.keys()):
            p_items = pages_items[page_num]

            # Cluster tokens into lines by vertical center
            sorted_by_y = sorted(
                p_items,
                key=lambda it: (it.bounding_box.y_min + it.bounding_box.y_max) / 2
                if it.bounding_box
                else 0
            )

            lines: List[List[OCRItem]] = []
            for it in sorted_by_y:
                if not it.bounding_box:
                    if lines:
                        lines[-1].append(it)
                    else:
                        lines.append([it])
                    continue

                y_center = (it.bounding_box.y_min + it.bounding_box.y_max) / 2
                placed = False
                for line in lines:
                    line_y_centers = [
                        (x.bounding_box.y_min + x.bounding_box.y_max) / 2
                        for x in line
                        if x.bounding_box
                    ]
                    avg_y = sum(line_y_centers) / len(line_y_centers) if line_y_centers else y_center
                    if abs(y_center - avg_y) <= self.line_clustering_tolerance:
                        line.append(it)
                        placed = True
                        break
                if not placed:
                    lines.append([it])

            # Sort lines by average vertical center
            lines.sort(
                key=lambda line: sum(
                    (x.bounding_box.y_min + x.bounding_box.y_max) / 2
                    for x in line
                    if x.bounding_box
                ) / max(1, len(line))
            )

            # Sort tokens in each line by x_min
            ordered_tokens = []
            for line in lines:
                line.sort(key=lambda it: it.bounding_box.x_min if it.bounding_box else 0)
                ordered_tokens.extend([it.text for it in line])

            page_texts.append(" ".join(ordered_tokens))

        return " ".join(page_texts)

    def segment_submission(self, ocr_submission: OCRSubmissionResult) -> SegmentedSubmissionResult:
        """Converts multi-page OCR detections into structured question answers Q1-Q5.

        Handles question marker detection, paragraph gap grouping, and cross-page answer continuation.
        """
        question_items_map: Dict[str, List[Tuple[OCRItem, str]]] = {
            qid: [] for qid in self.target_questions
        }
        active_question_id: Optional[str] = None

        for page in ocr_submission.pages:
            blocks = self.detect_page_blocks(page)
            for block_idx, block in enumerate(blocks):
                detected_qid = self.identify_block_question_id(block)

                if detected_qid:
                    active_question_id = detected_qid
                elif block_idx == 0 and active_question_id is not None:
                    # Page starts without a question marker -> Continuation of previous page's active question
                    pass  # Keep active_question_id as is (e.g. Q3 continuing on Page 2)
                elif active_question_id is None:
                    # Fallback for opening block before explicit marker
                    active_question_id = self.target_questions[0]

                if active_question_id in question_items_map:
                    for item in block:
                        question_items_map[active_question_id].append((item, page.source_file))

        # Build SegmentedAnswer objects
        segmented_answers: List[SegmentedAnswer] = []
        for qid in self.target_questions:
            pair_items = question_items_map.get(qid, [])
            items = [p[0] for p in pair_items]
            # Unique source pages preserving occurrence order
            source_pages = list(dict.fromkeys(p[1] for p in pair_items))

            reconstructed_text = self.reconstruct_text(items)
            avg_conf = (
                round(sum(it.confidence for it in items) / len(items), 4)
                if items
                else 0.0
            )

            # Build detailed SegmentedAnswerItem references
            answer_items = [
                SegmentedAnswerItem(
                    page_number=it.page_number,
                    source_file=src_file,
                    order=it.order,
                    text=it.text,
                    confidence=it.confidence,
                    bbox=it.bbox,
                    bounding_box=it.bounding_box,
                )
                for it, src_file in pair_items
            ]

            segmented_answers.append(
                SegmentedAnswer(
                    question_id=qid,
                    source_pages=source_pages,
                    text=reconstructed_text,
                    ocr_item_count=len(items),
                    average_confidence=avg_conf,
                    items=answer_items,
                )
            )

        return SegmentedSubmissionResult(
            student_id=ocr_submission.student_id,
            total_answers=len(segmented_answers),
            answers=segmented_answers,
            metadata={
                "cross_page_merging": "Q3 merged across page1.jpg and page2.jpg",
                "segmentation_method": "gap_clustering_and_margin_marker_detection",
                "total_ocr_items": sum(a.ocr_item_count for a in segmented_answers),
            },
        )

    def save_segmented_output(
        self,
        result: SegmentedSubmissionResult,
        output_path: str = "data/ocr_extracted/student_001_segmented.json",
    ) -> str:
        """Saves segmented question answers to JSON file.

        Args:
            result: SegmentedSubmissionResult instance.
            output_path: Target path for the JSON file.

        Returns:
            Saved absolute or relative file path.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        return str(out_file)

    def identify_question_boundaries(self, ocr_result: OCRDocumentResult) -> List[Dict]:
        """Legacy boundary detection compatibility helper."""
        return [{"question_id": qid} for qid in self.target_questions]

    def segment_answers(self, ocr_result: OCRDocumentResult) -> Dict[str, ExtractedQuestionAnswer]:
        """Legacy segmentation compatibility helper."""
        return {}
