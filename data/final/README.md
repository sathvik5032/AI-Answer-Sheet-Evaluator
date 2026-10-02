# Final Submission & Execution Guide: Automated Answer Sheet Evaluator

## 1. Project Purpose
This repository implements an end-to-end automated pipeline for evaluating handwritten student exam answer sheets. The system processes raw document images, performs OCR and line segmentation, semantically scores answers against fine-grained rubric criteria using transformer embeddings, detects factual contradictions regardless of keyword overlap, estimates evaluation confidence through multi-factor signals, and flags uncertain evaluations for human educator review.

## 2. Dataset Used
- **Subject**: Computer Fundamentals (Total Marks: 25.0, 5 Questions @ 5 Marks each).
- **Submission**: Single student submission (`student_001`) consisting of a two-page handwritten answer sheet.
- **Five Intended Evaluation Test Cases**:
  - **Q1**: Clearly correct answer with standard terminology $\rightarrow$ **5.0/5.0** (HIGH confidence).
  - **Q2**: Correct answer expressed with different wording and moderate OCR noise $\rightarrow$ **5.0/5.0** (HIGH confidence).
  - **Q3**: Correct answer spanning across Page 1 and Page 2 as a single continuous answer $\rightarrow$ **5.0/5.0** (HIGH confidence).
  - **Q4**: Incorrect meaning despite surface keyword overlap (keyboard asserted as an output device displaying text) $\rightarrow$ **0.0/5.0** (HIGH confidence due to explicit contradiction).
  - **Q5**: Correct answer containing extra non-essential information, but written with noisy, clumsy handwriting ($44\%$ low-confidence OCR tokens) $\rightarrow$ **5.0/5.0** (LOW confidence, flagged for human review).

## 3. Key File Locations
- **Question Paper**: `data/inputs/question_paper/Computer_Fundamentals_Question_Paper.pdf`
- **Handwritten Answer Sheets**:
  - Page 1: `data/inputs/answer_sheets/page1.jpg`
  - Page 2: `data/inputs/answer_sheets/page2.jpg`
- **Grading Rubric**: `data/rubrics/computer_fundamentals.json`
- **Rubric Schema**: `data/rubrics/rubric_schema.json`
- **Ground Truth Reference**: `data/references/ground_truth.json`
- **Source Metadata**: `data/references/source_metadata.json`

## 4. Final Generated Results
- **Final Evaluation JSON**: `data/final/student_001_final_evaluation.json`
- **Final Evaluation CSV**: `data/final/student_001_final_evaluation.csv`
- **Pipeline Approach Note**: `data/final/approach_note.md`
- **Intermediate Stage Artifacts**:
  - OCR Detections: `data/ocr_extracted/student_001_ocr.json`
  - Segmented Questions: `data/ocr_extracted/student_001_segmented.json`
  - Semantic Evaluations: `data/results/student_001_evaluation.json`
  - Confidence Scoring: `data/results/student_001_confidence.json`

## 5. How to Run the Pipeline

### Prerequisites
Activate the Python virtual environment and ensure dependencies are installed:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### End-to-End Master Execution
Run the complete pipeline from OCR through final reporting in a single command:
```bash
python scripts/run_pipeline.py
```

### Running Individual Stages
You can also execute each stage independently:
```bash
# 1. OCR Extraction (EasyOCR on page1.jpg + page2.jpg)
python scripts/run_ocr_pipeline.py

# 2. Question Segmentation (Boundary detection & cross-page continuation)
python scripts/run_segmentation.py

# 3. Semantic Evaluation (Sentence embeddings & contradiction detection)
python scripts/run_evaluation.py

# 4. Confidence Scoring (Multi-factor evaluation confidence)
python scripts/run_confidence.py

# 5. Final Report Generation (JSON and CSV packaging in data/final/)
python scripts/generate_final_reports.py
```

### Running the Test Suite
Execute the full test suite across all stages:
```bash
python -m pytest -v tests/
```

## 6. Final Evaluation Summary (20.0 / 25.0)

| Question | Max | Score Awarded | Confidence | Human Review | Score vs. Key Evidence Summary |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1** | 5.0 | **5.0** | **HIGH** | `false` | All 5 criteria satisfied: electronic device, input data, process, output info, desktop example. |
| **Q2** | 5.0 | **5.0** | **HIGH** | `false` | All 4 criteria satisfied: RAM memory, temporary/volatile, active programs, lost when switched off. |
| **Q3** | 5.0 | **5.0** | **HIGH** | `false` | All 5 criteria satisfied across Page 1 & 2: system software, user interface, hardware management, run programs, Windows example. |
| **Q4** | 5.0 | **0.0** | **HIGH** | `false` | Explicit contradiction: asserted keyboard is an output device displaying text; 0 marks awarded despite keyword presence. |
| **Q5** | 5.0 | **5.0** | **LOW** | **`true`** | All 4 criteria satisfied: browser software, view websites, internet access, Chrome example. Extra info preserved without penalty. |
| **TOTAL**| **25.0**| **20.0** | — | **1 question (`Q5`)** | Ground-truth test-case alignment: 5/5 question outcomes matched the expected evaluation behavior; final score: 20/25. |

## 7. Why Q5 is Flagged for Human Review
While Q5 semantically satisfies all rubric criteria (earning 5.0/5.0), its underlying handwritten evidence is severely degraded:
- **43.75% (14 of 32)** of detected OCR items have extraction confidence below $0.50$.
- Mean OCR confidence is depressed at $0.6098$.
- Multiple tokens suffer heavy distortion (`(haone`, `olen`, `webs (tes`, `4 {5`, `80 Forma tn`).
- Rather than falsely claiming high certainty or penalizing the student's score, the system distinguishes **evaluation confidence** from **awarded score**: it awards full credit based on available semantic signals, but routes the question to the human educator review queue (`human_review_required = true`) to confirm the handwritten text.

## 8. Known Limitations & Operational Boundaries
- **OCR Quality on Noisy Handwriting**: The system does not claim perfect OCR. Severely distorted characters and irregular cursive handwriting introduce uncertainty, which is why multi-factor confidence scoring automatically flags uncertain evaluations for human verification.
- **Domain Scope**: Semantic evaluation and contradiction detection are calibrated for Computer Fundamentals. Expanding to other subjects requires registering appropriate domain rubrics, vocabulary, and opposition pairings.
- **Assistive System Design**: This pipeline is designed as an assistive grading tool to accelerate educator evaluation, not an infallible or universally autonomous decision-maker.
