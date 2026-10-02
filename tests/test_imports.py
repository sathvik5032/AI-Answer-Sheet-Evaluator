import os
import sys

# Prevent OpenMP runtime conflict and network timeouts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_required_imports():
    modules = [
        "numpy",
        "yaml",
        "pydantic",
        "pandas",
        "sklearn",
        "PIL",
        "cv2",
        "fitz",
        "easyocr",
        "sentence_transformers",
    ]
    failed = []
    for mod_name in modules:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "loaded")
            print(f"[OK] {mod_name:22} : {ver}", flush=True)
        except Exception as e:
            print(f"[FAIL] {mod_name:20} : {e}", flush=True)
            failed.append(mod_name)

    if failed:
        sys.exit(1)
    print("\nAll required imports succeeded!", flush=True)


if __name__ == "__main__":
    test_required_imports()
