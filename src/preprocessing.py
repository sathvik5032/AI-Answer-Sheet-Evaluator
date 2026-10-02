"""Preprocessing module for handwritten answer sheet images and PDFs.

Prepares raw scans and photos by loading formats (PDF, JPG, PNG), detecting and
correcting minor rotational skew, and normalizing contrast to optimize handwriting
recognition without over-processing.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class DocumentPreprocessor:
    """Handles image and PDF loading, normalization, and visual preprocessing."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.target_dpi = self.config.get("target_dpi", 300)
        self.enable_deskew = self.config.get("enable_deskew", True)
        self.enable_denoise = self.config.get("enable_denoise", False)
        self.contrast_factor = self.config.get("contrast_factor", 1.2)
        self.convert_grayscale = self.config.get("convert_grayscale", False)

    def load_document(self, file_path: str) -> List[np.ndarray]:
        """Loads an image (PNG, JPG) or multi-page PDF into an in-memory image list.

        Args:
            file_path: Path to the input document image or PDF.

        Returns:
            List of loaded image objects as BGR NumPy arrays (one per page).
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            if fitz is None:
                raise ImportError("PyMuPDF (fitz) is required to load PDF documents.")
            doc = fitz.open(str(path))
            pages = []
            zoom = self.target_dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                # Convert pixmap samples to numpy BGR array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pages.append(img_bgr)
            doc.close()
            return pages

        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            img = cv2.imread(str(path))
            if img is None:
                # Fallback to PIL in case of non-standard headers
                with Image.open(str(path)) as pil_img:
                    pil_rgb = pil_img.convert("RGB")
                    img = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
            return [img]

        else:
            raise ValueError(f"Unsupported document format: {ext}")

    def detect_skew_angle(self, image: np.ndarray, max_angle: float = 15.0) -> float:
        """Estimates rotational skew angle in degrees using text line orientation.

        Args:
            image: Input image (BGR or grayscale).
            max_angle: Maximum allowable angle to consider.

        Returns:
            Estimated skew angle in degrees.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert: dark ink becomes bright foreground
        inv = cv2.bitwise_not(gray)

        # Threshold to keep prominent ink strokes
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Use non-zero pixels coordinates
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return 0.0

        # Calculate minimum bounding rectangle
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # OpenCV minAreaRect returns angle in [-90, 0)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > max_angle:
            return 0.0

        return float(angle)

    def deskew(self, image: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
        """Detects and corrects rotational skew in scanned answer pages.

        Args:
            image: Input image object (BGR).
            max_angle: Maximum rotation angle considered valid.

        Returns:
            Tuple of (deskewed_image, angle_applied_degrees).
        """
        angle = self.detect_skew_angle(image, max_angle=max_angle)

        # Skip rotation if angle is negligibly small to avoid interpolation softening
        if abs(angle) < 0.5:
            return image, 0.0

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed, float(angle)

    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Applies gentle adaptive contrast normalization (CLAHE) to preserve ink stroke detail.

        Args:
            image: Input image object (BGR or grayscale).
            clip_limit: Contrast threshold limit for CLAHE.

        Returns:
            Contrast-enhanced image object.
        """
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE strictly on the Lightness channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Applies edge-preserving bilateral filter to suppress paper texture noise.

        Args:
            image: Input image object (BGR).

        Returns:
            Denoised image.
        """
        return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies configured preprocessing steps to a single image array.

        Args:
            image: Raw image array (BGR).

        Returns:
            Tuple of (preprocessed_image, applied_metadata).
        """
        metadata: Dict[str, Any] = {
            "original_shape": list(image.shape),
            "deskew_angle": 0.0,
            "contrast_enhanced": False,
            "denoised": False,
        }
        processed = image.copy()

        if self.enable_deskew:
            processed, angle = self.deskew(processed)
            metadata["deskew_angle"] = angle

        if self.enable_denoise:
            processed = self.denoise(processed)
            metadata["denoised"] = True

        if self.contrast_factor > 1.0:
            processed = self.enhance_contrast(processed, clip_limit=self.contrast_factor)
            metadata["contrast_enhanced"] = True

        if self.convert_grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            metadata["grayscale"] = True

        metadata["final_shape"] = list(processed.shape)
        return processed, metadata

    def process(self, file_path: str) -> List[np.ndarray]:
        """Executes full preprocessing pipeline for a given document file.

        Args:
            file_path: Path to the document image or PDF.

        Returns:
            List of preprocessed image pages ready for OCR.
        """
        raw_pages = self.load_document(file_path)
        processed_pages = []
        for page in raw_pages:
            proc, _ = self.preprocess_image(page)
            processed_pages.append(proc)
        return processed_pages
