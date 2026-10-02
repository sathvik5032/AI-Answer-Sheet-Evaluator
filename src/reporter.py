"""Result Generation and Reporting module.

Transforms structured evaluation records into final standardized JSON and tabular CSV
reports, summarizing question-by-question marks, similarity, confidence, evidence, and human-review flags.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import (
    FinalEvaluationReport,
    MasterRubric,
    QuestionConfidenceResult,
    QuestionEvaluation,
    SegmentedSubmissionResult,
    StudentConfidenceResult,
    StudentEvaluationResult,
)


class ResultGenerator:
    """Exports pipeline evaluation outputs into persistent JSON and CSV formats."""

    def __init__(self, output_dir: Optional[str] = "data/final"):
        self.output_dir = Path(output_dir or "data/final")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, data: Any, file_name: str = "student_001_final_evaluation.json") -> str:
        """Exports data to a structured JSON file."""
        out_path = self.output_dir / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            if hasattr(data, "model_dump_json"):
                f.write(data.model_dump_json(indent=2))
            else:
                json.dump(data, f, indent=2)
        return str(out_path)

    def export_csv(self, records: List[Dict[str, Any]], file_name: str = "student_001_final_evaluation.csv") -> str:
        """Exports tabular records to CSV format."""
        out_path = self.output_dir / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            return str(out_path)

        fieldnames = list(records[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        return str(out_path)

    def generate_all(self, report: FinalEvaluationReport, base_name: str = "evaluation_report") -> dict:
        """Generates legacy report exports for backward compatibility."""
        json_path = self.export_json(report, f"{base_name}.json")
        csv_records = [
            {
                "question_id": r.question_id,
                "awarded_marks": r.awarded_marks,
                "max_marks": r.max_marks,
                "similarity_score": r.similarity_score,
                "confidence_level": r.confidence_level,
                "confidence_reason": r.confidence_reason,
            }
            for r in report.results
        ]
        csv_path = self.export_csv(csv_records, f"{base_name}.csv")
        return {"json": json_path, "csv": csv_path}

    def generate_final_submission_reports(
        self,
        eval_path: str = "data/results/student_001_evaluation.json",
        conf_path: str = "data/results/student_001_confidence.json",
        seg_path: str = "data/ocr_extracted/student_001_segmented.json",
        rubric_path: str = "data/rubrics/computer_fundamentals.json",
    ) -> Dict[str, str]:
        """Generates both final JSON and CSV reports for student_001."""
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        with open(conf_path, "r", encoding="utf-8") as f:
            conf_data = json.load(f)
        with open(seg_path, "r", encoding="utf-8") as f:
            seg_data = json.load(f)
        with open(rubric_path, "r", encoding="utf-8") as f:
            rubric_data = json.load(f)

        student_eval = StudentEvaluationResult(**eval_data)
        student_conf = StudentConfidenceResult(**conf_data)
        student_seg = SegmentedSubmissionResult(**seg_data)
        master_rubric = MasterRubric(**rubric_data)

        seg_map = {a.question_id: a for a in student_seg.answers}
        conf_map = {q.question_id: q for q in student_conf.questions}
        rubric_map = {q.question_id: q for q in master_rubric.questions}

        json_questions = []
        csv_records = []

        for qe in student_eval.questions:
            qid = qe.question_id
            seg_ans = seg_map.get(qid)
            q_conf = conf_map.get(qid)
            q_rubric = rubric_map.get(qid)

            extracted_text = seg_ans.text if seg_ans else ""
            score = qe.score
            max_score = qe.max_score
            conf_label = q_conf.confidence.label if q_conf else "MEDIUM"
            conf_reason = q_conf.confidence.reason if q_conf else ""
            human_review = q_conf.human_review_required if q_conf else False

            # Synthesize score_vs_key summary
            contra = next((c for c in qe.criterion_evaluations if c.status == "contradicted"), None)
            if contra:
                score_vs_key = (
                    f"0/{max_score} marks: Factually contradictory assertion detected. "
                    f"Student asserted opposing classification ({contra.reason}) contrary to rubric requirement."
                )
            else:
                sat_count = sum(1 for c in qe.criterion_evaluations if c.status == "satisfied")
                total_crit = len(qe.criterion_evaluations)
                crit_summaries = "; ".join(f"{c.criterion_id}: {c.status} ({c.marks_awarded}m)" for c in qe.criterion_evaluations)
                score_vs_key = f"{score}/{max_score} marks: {sat_count}/{total_crit} rubric criteria satisfied [{crit_summaries}]."

            csv_records.append({
                "question_id": qid,
                "extracted_answer": extracted_text,
                "score": score,
                "max_score": max_score,
                "score_vs_key": score_vs_key,
                "confidence": conf_label,
                "reason": conf_reason,
                "human_review_required": human_review,
            })

            json_questions.append({
                "question_id": qid,
                "question_prompt": q_rubric.prompt if q_rubric else "",
                "model_answer": q_rubric.model_answer if q_rubric else "",
                "extracted_answer": extracted_text,
                "score": score,
                "max_score": max_score,
                "score_vs_key": score_vs_key,
                "confidence": conf_label,
                "reason": conf_reason,
                "human_review_required": human_review,
                "criterion_evaluations": [c.model_dump() for c in qe.criterion_evaluations],
                "confidence_factors": q_conf.confidence.factors if q_conf else {},
            })

        final_json_data = {
            "student_id": student_eval.student_id,
            "subject": student_eval.subject,
            "total_score": student_eval.total_score,
            "total_max_score": student_eval.total_max_score,
            "percentage": round((student_eval.total_score / student_eval.total_max_score) * 100, 2),
            "human_review_required": len(student_conf.human_review_questions) > 0,
            "human_review_questions": student_conf.human_review_questions,
            "summary": {
                "total_questions": len(json_questions),
                "high_confidence_count": sum(1 for q in json_questions if q["confidence"] == "HIGH"),
                "medium_confidence_count": sum(1 for q in json_questions if q["confidence"] == "MEDIUM"),
                "low_confidence_count": sum(1 for q in json_questions if q["confidence"] == "LOW"),
                "human_review_required_count": sum(1 for q in json_questions if q["human_review_required"]),
            },
            "questions": json_questions,
        }

        json_file = self.export_json(final_json_data, "student_001_final_evaluation.json")
        csv_file = self.export_csv(csv_records, "student_001_final_evaluation.csv")

        return {
            "json_report": json_file,
            "csv_report": csv_file,
        }
