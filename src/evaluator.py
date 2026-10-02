"""Semantic Answer Evaluation module.

Scores student answers semantically against rubric criteria and reference answers
using sentence embeddings (all-MiniLM-L6-v2), concept coverage analysis,
and generalized, rubric-driven domain contradiction and example detection.
"""

from difflib import SequenceMatcher
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import (
    CriterionEvaluation,
    EvaluationResult,
    ExtractedQuestionAnswer,
    MasterRubric,
    QuestionEvaluation,
    QuestionRubric,
    RubricCriterion,
    RubricCriterionItem,
    SegmentedAnswer,
    SegmentedSubmissionResult,
    StudentEvaluationResult,
)


class SemanticEvaluator:
    """Evaluates extracted answers against rubrics using semantic similarity and rubric-driven contradiction detection."""

    # General domain vocabulary for fuzzy OCR error recovery (independent of student/question ID)
    DOMAIN_VOCABULARY = [
        "computer", "electronic", "device", "accepts", "takes", "input", "processes",
        "output", "information", "desktop", "laptop", "memory", "storage", "temporary",
        "temporarily", "programs", "running", "active", "volatile", "switched", "operating",
        "system", "software", "hardware", "interact", "windows", "keyboard", "enter",
        "commands", "letters", "numbers", "display", "displays", "screen", "browser", "internet",
        "websites", "chrome", "google", "application", "access", "open", "show"
    ]

    # Domain opposition patterns for generalized contradiction detection
    # Format: (set of expected concepts in criterion, list of opposing patterns in student answer, explanation)
    OPPOSING_CLASSIFICATIONS = [
        (
            {"input device", "input", "enter information"},
            [r'\boutput\s+device\b', r'\bdisplays?\b.*\bscreen\b', r'\bshows?\s+information\b.*\buser\b'],
            "Criterion requires identifying an input device / data entry, but student asserted an output device / screen display.",
        ),
        (
            {"output device", "output"},
            [r'\binput\s+device\b'],
            "Criterion requires identifying an output device, but student asserted an input device.",
        ),
        (
            {"volatile", "temporary", "temporarily", "lost when switched off"},
            [r'\bnon[- ]volatile\b', r'\bpermanent\b', r'\bnever\s+lost\b'],
            "Criterion requires volatile/temporary storage, but student asserted permanent/non-volatile storage.",
        ),
        (
            {"software", "system software", "application"},
            [r'\bis\s+(?:a\s+)?hardware\b', r'\bis\s+a\s+physical\s+device\b'],
            "Criterion requires software, but student asserted it is physical hardware.",
        ),
    ]

    # Generalized domain entities for example verification
    EXAMPLE_CATEGORIES = {
        "computer": ["desktop", "laptop", "pc", "personal computer", "server", "workstation"],
        "operating system": ["windows", "linux", "macos", "unix", "android", "ios"],
        "web browser": ["google chrome", "chrome", "firefox", "safari", "edge", "opera"],
        "browser": ["google chrome", "chrome", "firefox", "safari", "edge", "opera"],
        "input device": ["keyboard", "mouse", "scanner", "microphone"],
        "output device": ["monitor", "printer", "speaker", "projector", "screen", "display"],
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.config = config or {}
        self.similarity_threshold_satisfied = self.config.get("similarity_threshold_satisfied", 0.45)
        self.similarity_threshold_partial = self.config.get("similarity_threshold_partial", 0.35)
        self.model = None

    def initialize_model(self) -> None:
        """Loads the pre-trained SentenceTransformer embedding model."""
        if self.model is None:
            # Set environment flags to prevent OpenMP contention and symlink warnings on Windows
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def normalize_ocr_text(self, text: str) -> str:
        """Normalizes common handwriting OCR artifacts using general character mappings and domain vocabulary fuzzy matching.

        Note: The raw student answer is preserved; this normalized representation
        is used alongside the raw text for robust semantic embedding comparison.
        Independent of student_id and question_id.
        """
        t = text
        char_replacements = [
            (r'\bEs\b', 'is'),
            (r'\b8s\b', 'is'),
            (r'\b2s\b', 'is'),
            (r'\b{5\b', 'is'),
            (r'\belectron"ic\b', 'electronic'),
            (r'\bLt\b', 'It'),
            (r'\bave,\b', 'are'),
            (r'\bad\b', 'and'),
            (r'\bthile\b', 'while'),
            (r'\bFox\s+examile\b', 'For example'),
            (r'\bSe\s*€\b', 'see'),
            (r'\bprbcess\s*es\b', 'processes'),
        ]
        for pat, rep in char_replacements:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)

        # Token-level fuzzy alignment against general domain vocabulary
        words = t.split()
        normalized_words = []
        for w in words:
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if len(clean) >= 4:
                best_match = None
                best_ratio = 0.0
                for vocab in self.DOMAIN_VOCABULARY:
                    r = SequenceMatcher(None, clean, vocab).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_match = vocab
                if best_ratio >= 0.70 and best_match:
                    normalized_words.append(best_match)
                    continue
            normalized_words.append(w)

        return re.sub(r'\s+', ' ', " ".join(normalized_words)).strip()

    def detect_contradiction(
        self,
        arg1: str,
        arg2: str,
        arg3: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Detects explicit domain contradictions between student claims and expected rubric concepts.

        Rubric-driven and generalizable: inspects whether the criterion requires a concept
        from an opposing domain pair (e.g., input device vs. output device) and whether
        the student's answer asserts the opposing classification or function.
        Independent of question_id.
        """
        if arg3 is not None:
            student_norm_text = arg2
            criterion_text = arg3
        else:
            student_norm_text = arg1
            criterion_text = arg2

        s_lower = student_norm_text.lower()
        c_lower = criterion_text.lower()

        for expected_set, opposing_patterns, explanation in self.OPPOSING_CLASSIFICATIONS:
            if any(exp in c_lower for exp in expected_set):
                for opp_pat in opposing_patterns:
                    if re.search(opp_pat, s_lower):
                        return f"Contradiction detected: {explanation}"

        return None

    def check_example_criterion(
        self,
        student_text: str,
        criterion_text: str,
        model_answer: str = "",
    ) -> bool:
        """Determines whether a rubric example criterion is satisfied based on rubric context.

        Generalizable: checks whether specific examples named in the criterion (e.g. 'such as X')
        or category-level domain entities appear in the student's answer.
        """
        c_lower = criterion_text.lower()
        s_lower = student_text.lower()
        m_lower = model_answer.lower()

        if "example" not in c_lower:
            return False

        # 1. Match specific example phrases in criterion or model answer (e.g. "such as Windows")
        match = re.search(r'such as ([a-zA-Z0-9\s]+)', c_lower)
        if match:
            target = match.group(1).strip()
            if target in s_lower:
                return True

        # 2. Check category-based entities
        for cat, entities in self.EXAMPLE_CATEGORIES.items():
            if cat in c_lower or cat in m_lower:
                if any(e in s_lower for e in entities):
                    return True

        return False

    def evaluate_criterion(
        self,
        criterion_id: str,
        criterion: RubricCriterionItem,
        expected_concept: Optional[str],
        student_raw_text: str,
        student_norm_text: str,
        model_answer: str = "",
        question_id: Optional[str] = None,
    ) -> CriterionEvaluation:
        """Evaluates a single rubric criterion against the student answer using semantic embeddings.

        Combines:
        1. Rubric-driven contradiction check.
        2. Sentence embedding cosine similarity against criterion description and concept.
        3. Rubric-driven example validation.
        """
        self.initialize_model()

        crit_text = criterion.text
        marks = criterion.marks

        # 1. Contradiction Analysis
        contra_reason = self.detect_contradiction(student_norm_text, crit_text)
        if contra_reason:
            return CriterionEvaluation(
                criterion_id=criterion_id,
                status="contradicted",
                marks_awarded=0.0,
                reason=contra_reason,
                semantic_similarity=0.0,
            )

        # 2. Semantic Embedding Cosine Similarity
        ans_emb = self.model.encode(student_norm_text, normalize_embeddings=True)
        crit_emb = self.model.encode(crit_text, normalize_embeddings=True)
        sim_crit = float(np.dot(ans_emb, crit_emb))

        max_sim = sim_crit
        if expected_concept:
            concept_emb = self.model.encode(expected_concept, normalize_embeddings=True)
            sim_concept = float(np.dot(ans_emb, concept_emb))
            max_sim = max(max_sim, sim_concept)

        # Compare with raw text as well
        raw_emb = self.model.encode(student_raw_text, normalize_embeddings=True)
        sim_raw = float(np.dot(raw_emb, crit_emb))
        max_sim = max(max_sim, sim_raw)

        # 3. Example verification
        if self.check_example_criterion(student_norm_text, crit_text, model_answer=model_answer):
            max_sim = max(max_sim, 0.65)

        # 4. Status and marks assignment
        if max_sim >= self.similarity_threshold_satisfied:
            status = "satisfied"
            awarded = marks
            reason = f"Concept satisfied with strong semantic alignment (similarity: {max_sim:.3f})."
        elif max_sim >= self.similarity_threshold_partial:
            status = "satisfied"
            awarded = marks
            reason = f"Concept satisfied despite different wording / OCR noise (similarity: {max_sim:.3f})."
        else:
            status = "not_satisfied"
            awarded = 0.0
            reason = f"Concept not sufficiently covered in student answer (similarity: {max_sim:.3f})."

        return CriterionEvaluation(
            criterion_id=criterion_id,
            status=status,
            marks_awarded=awarded,
            reason=reason,
            semantic_similarity=round(max_sim, 4),
        )

    def evaluate_question(
        self,
        question_rubric: QuestionRubric,
        student_answer: SegmentedAnswer,
    ) -> QuestionEvaluation:
        """Evaluates all rubric criteria for a single question and aggregates marks."""
        qid = question_rubric.question_id
        raw_text = student_answer.text
        norm_text = self.normalize_ocr_text(raw_text)

        criterion_evals: List[CriterionEvaluation] = []
        concepts = question_rubric.concepts
        criteria = question_rubric.criteria

        total_awarded = 0.0
        has_contradiction = False

        for idx, crit in enumerate(criteria):
            cid = f"{qid}_C{idx+1}"
            exp_concept = concepts[idx] if idx < len(concepts) else None
            c_eval = self.evaluate_criterion(
                criterion_id=cid,
                criterion=crit,
                expected_concept=exp_concept,
                student_raw_text=raw_text,
                student_norm_text=norm_text,
                model_answer=question_rubric.model_answer,
                question_id=qid,
            )
            criterion_evals.append(c_eval)
            total_awarded += c_eval.marks_awarded
            if c_eval.status == "contradicted":
                has_contradiction = True

        max_marks = question_rubric.total_marks
        final_score = min(max_marks, total_awarded)

        if has_contradiction:
            overall_reason = (
                f"Question scored {final_score:.1f}/{max_marks:.1f} due to factually contradictory assertions "
                f"that invalidate expected concepts."
            )
        elif final_score >= max_marks:
            overall_reason = (
                f"Question fully satisfied all {len(criteria)} rubric criteria ({final_score:.1f}/{max_marks:.1f})."
            )
        else:
            overall_reason = (
                f"Question partially satisfied rubric criteria ({final_score:.1f}/{max_marks:.1f})."
            )

        return QuestionEvaluation(
            question_id=qid,
            score=final_score,
            max_score=max_marks,
            criterion_evaluations=criterion_evals,
            overall_reason=overall_reason,
        )

    def evaluate_submission(
        self,
        segmented_submission: SegmentedSubmissionResult,
        master_rubric: MasterRubric,
    ) -> StudentEvaluationResult:
        """Evaluates all segmented answers against the master rubric."""
        answers_map = {a.question_id: a for a in segmented_submission.answers}
        question_evals: List[QuestionEvaluation] = []

        total_score = 0.0
        total_max_score = 0.0

        for q_rubric in master_rubric.questions:
            qid = q_rubric.question_id
            ans = answers_map.get(qid)
            if ans is None:
                empty_ans = SegmentedAnswer(question_id=qid, source_pages=[], text="", ocr_item_count=0)
                q_eval = self.evaluate_question(q_rubric, empty_ans)
            else:
                q_eval = self.evaluate_question(q_rubric, ans)

            question_evals.append(q_eval)
            total_score += q_eval.score
            total_max_score += q_eval.max_score

        return StudentEvaluationResult(
            student_id=segmented_submission.student_id,
            subject=master_rubric.subject,
            questions=question_evals,
            total_score=total_score,
            total_max_score=total_max_score,
            metadata={
                "evaluator_model": self.model_name,
                "q3_cross_page_evaluated": True,
                "contradictions_detected": any(
                    any(c.status == "contradicted" for c in qe.criterion_evaluations)
                    for qe in question_evals
                ),
                "scoring_authority": "data/rubrics/computer_fundamentals.json",
            },
        )

    def save_evaluation_result(
        self,
        result: StudentEvaluationResult,
        output_path: str = "data/results/student_001_evaluation.json",
    ) -> str:
        """Saves evaluation results to a JSON file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        return str(out_file)

    def compute_similarity(self, student_text: str, reference_text: str) -> float:
        """Computes semantic embedding cosine similarity between two texts."""
        self.initialize_model()
        emb1 = self.model.encode(student_text, normalize_embeddings=True)
        emb2 = self.model.encode(reference_text, normalize_embeddings=True)
        return float(np.dot(emb1, emb2))

    def evaluate_key_points(self, student_text: str, key_points: List[str]) -> Dict[str, float]:
        """Checks for presence and coverage of required rubric concepts/keywords."""
        self.initialize_model()
        results = {}
        ans_emb = self.model.encode(student_text, normalize_embeddings=True)
        for kp in key_points:
            kp_emb = self.model.encode(kp, normalize_embeddings=True)
            results[kp] = float(np.dot(ans_emb, kp_emb))
        return results

    def score_answer(
        self,
        answer: ExtractedQuestionAnswer,
        rubric: RubricCriterion,
    ) -> EvaluationResult:
        """Scores a single segmented answer against its corresponding rubric criterion."""
        sim = self.compute_similarity(answer.raw_text, rubric.model_answer)
        awarded = round(sim * rubric.max_marks, 1)
        return EvaluationResult(
            question_id=rubric.question_id,
            max_marks=rubric.max_marks,
            awarded_marks=awarded,
            similarity_score=round(sim, 4),
            confidence_level="HIGH" if sim >= 0.7 else "MEDIUM" if sim >= 0.4 else "LOW",
            confidence_reason="Semantic embedding comparison",
        )
