# AI/ML Engineer — Answer Sheet Evaluation Pipeline

**Subject Focus**: Computer Fundamentals  
**Pipeline Type**: Automated Handwritten Answer Sheet Evaluation (OCR/HTR + Semantic Scoring)

---

## 1. Overview & Objective

This project implements an end-to-end automated pipeline for evaluating handwritten student answer sheets against an established answer key and grading rubric.

The complete pipeline achieves 5 key objectives:
1. **Extraction**: Ingest handwritten answer sheet images or multi-page PDFs and extract text and confidence values using pre-trained OCR/HTR models (e.g., EasyOCR).
2. **Segmentation**: Identify question delimiters (e.g., `Q1.`, `Ans 2`, `3)`) to separate continuous text into structured per-question answer blocks.
3. **Semantic Scoring**: Evaluate each student answer against a Computer Fundamentals rubric and model answer using sentence embeddings (e.g., `sentence-transformers`) and key-concept coverage.
4. **Confidence Scoring**: Assign a multi-factor confidence rating (`HIGH`, `MEDIUM`, `LOW`) along with an explicit human-interpretable reason.
5. **Output Generation**: Produce structured, exportable evaluation reports in JSON and CSV formats.

---

## 2. Planned Pipeline Architecture

```text
[ Handwritten Answer Sheet (Image / PDF) ]
                     │
                     ▼
       ┌───────────────────────────┐
       │ 1. Document Preprocessing │  (Orientation, Deskew, Contrast Normalization)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │  2. OCR / HTR Extraction  │  (Pre-trained EasyOCR, Token Confidences)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ 3. Question Segmentation  │  (Delimiter Detection & Grouping by Q#)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │   4. Semantic Scoring     │ ◄── [ Answer Key & Rubric (Computer Fundamentals) ]
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │   5. Confidence Scorer    │  (OCR Quality + Semantic Alignment + Completeness)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ 6. Reporting & Exporting  │
       └─────────────┬─────────────┘
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ evaluation_report.json ]    [ evaluation_report.csv ]
```

---

## 3. Project Structure

```text
Evaluator/
├── .gitignore                      # Version control exclusion rules
├── .env.example                    # Optional API credentials template
├── README.md                       # Architecture & execution guide
├── requirements.txt                # Curated Python dependencies
├── main.py                         # CLI pipeline orchestrator entrypoint
├── config/
│   └── config.yaml                 # Pipeline hyperparameters & directory mappings
├── data/
│   ├── inputs/
│   │   ├── question_paper/         # Source question paper documents
│   │   └── answer_sheets/          # Scanned / photographed handwritten answer sheets
│   ├── rubrics/                    # Master answer keys and grading criteria
│   ├── ocr_extracted/              # Persisted raw and structured OCR/HTR outputs
│   ├── results/                    # Generated JSON and CSV grading sheets
│   └── references/                 # Syllabus notes, textbook excerpts, reference materials
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration loader
│   ├── models.py                   # Pydantic data schemas (OCR, answers, rubrics, results)
│   ├── preprocessing.py            # Image/PDF deskew, denoise, and enhancement stubs
│   ├── ocr_engine.py               # Pre-trained OCR/HTR wrapper stubs
│   ├── segmentation.py             # Question boundary parser stubs
│   ├── evaluator.py                # Semantic similarity & rubric scoring stubs
│   ├── confidence.py               # Multi-factor confidence rating stubs
│   └── reporter.py                 # JSON & CSV report generator stubs
└── tests/
    ├── __init__.py
    └── test_pipeline_smoke.py      # Verification tests for schemas and class initializations
```

---

## 4. Expected Input Formats & Dataset Organization

To evaluate a handwritten student submission, the pipeline requires four coordinated inputs:

### A. Question Paper (`data/inputs/question_paper/`)
* **Supported Formats**: Multi-page or single-page PDF (`.pdf`), high-resolution images (`.png`, `.jpg`, `.jpeg`).
* **Content**: 3 to 5 Computer Fundamentals questions (e.g., hardware architecture, memory hierarchy, OS fundamentals).
* **Specifications**: Clear printed or typed text, standard 300 DPI recommended.

### B. Handwritten Student Answer Sheet (`data/inputs/answer_sheets/`)
* **Supported Formats**: Scanned or photographed images (`.png`, `.jpg`, `.jpeg`) or multi-page PDF (`.pdf`).
* **Layout Requirements**:
  * Clear question/answer markers at the beginning of each response (e.g., `Q1.`, `Ans 1.`, `1)`).
  * Good contrast between ink and page (blue or black ink on white/ruled paper).
  * Minimal rotational skew (< 2 degrees) and uniform illumination.

### C. Answer Key & Scoring Rubric (`data/rubrics/computer_fundamentals.json`)
Formatted as structured JSON validated against `data/rubrics/rubric_schema.json`:
* `subject`: Subject name (e.g., `"Computer Fundamentals"`).
* `total_marks`: Cumulative maximum marks across all questions.
* `questions`: List of question rubric objects, each specifying:
  * `question_id`: Unique question identifier (e.g., `"Q1"`).
  * `question_text`: Complete question statement.
  * `maximum_marks`: Total point value (e.g., `5.0`).
  * `expected_answer`: Comprehensive reference model answer.
  * `key_concepts`: Array of critical technical terms/concepts required.
  * `rubric_criteria`: Array of scoring criteria with individual point allocations (`criterion_id`, `description`, `marks`).

### D. Dataset Provenance & Metadata (`data/references/source_metadata.json`)
Tracks document provenance, source URL or citation, subject, number of questions, layout details, handwriting characteristics, and image quality assessment.

---

## 5. Pipeline Stages & Module Responsibilities

| Stage | Module | Primary Class | Responsibility |
|---|---|---|---|
| **0. Schemas & Models** | `src/models.py` | `OCRDocumentResult`, `RubricCriterion`, `EvaluationResult` | Typed contracts ensuring clean hand-offs between pipeline stages. |
| **1. Preprocessing** | `src/preprocessing.py` | `DocumentPreprocessor` | Loads images/PDFs, corrects rotational skew, and normalizes contrast for handwriting. |
| **2. OCR / HTR** | `src/ocr_engine.py` | `EasyOCREngine` | Uses pre-trained OCR to extract line-level text and associated token confidences. |
| **3. Segmentation** | `src/segmentation.py` | `QuestionSegmenter` | Groups extracted lines into structured answers mapped to each question number. |
| **4. Semantic Scoring**| `src/evaluator.py` | `SemanticEvaluator` | Embeds text with `sentence-transformers` and grades answers against rubric key points. |
| **5. Confidence** | `src/confidence.py` | `ConfidenceScorer` | Synthesizes OCR quality, semantic match, and length into `HIGH`/`MEDIUM`/`LOW` + reason. |
| **6. Reporting** | `src/reporter.py` | `ResultGenerator` | Exports comprehensive results to standardized `JSON` and tabular `CSV` files. |

---

## 6. Local Setup & Execution

### Prerequisites & Python Assumptions
- **Python Version**: Recommended `Python 3.10` or `Python 3.11` (64-bit).
- **Environment**: Virtual environment (`venv` or `conda`).

### Step 1: Create and Activate Virtual Environment
```powershell
# From project root: d:\Evaluator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the End-to-End Pipeline
```powershell
python scripts/run_pipeline.py
```
This executes all stages (OCR extraction, question segmentation, semantic evaluation, confidence scoring, and final report generation) and writes final artifacts to `data/final/`:
- Final Evaluation JSON: `data/final/student_001_final_evaluation.json`
- Final Evaluation CSV: `data/final/student_001_final_evaluation.csv`
- Approach Note: `data/final/approach_note.md`
- Submission Guide: `data/final/README.md`

### Step 4: Run Tests
```powershell
python -m pytest -v tests/
```
All 55 tests validate dataset integrity, OCR extraction, cross-page segmentation, generalized semantic evaluation, multi-factor confidence scoring, and final reporting.
