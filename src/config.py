"""Configuration manager for the evaluation pipeline."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Loads and exposes pipeline configuration settings."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.settings: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration file if it exists, otherwise provides defaults."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Fallback configuration if config file is not found."""
        return {
            "paths": {
                "question_paper_dir": "data/inputs/question_paper",
                "answer_sheets_dir": "data/inputs/answer_sheets",
                "rubrics_dir": "data/rubrics",
                "ocr_extracted_dir": "data/ocr_extracted",
                "results_dir": "data/results",
                "references_dir": "data/references",
            },
            "preprocessing": {
                "target_dpi": 300,
                "convert_grayscale": True,
                "enable_deskew": True,
            },
            "ocr": {
                "engine": "easyocr",
                "languages": ["en"],
                "use_gpu": False,
            },
            "confidence": {
                "thresholds": {"high": 0.75, "medium": 0.50}
            },
            "reporting": {
                "export_formats": ["json", "csv"]
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration key."""
        return self.settings.get(key, default)
