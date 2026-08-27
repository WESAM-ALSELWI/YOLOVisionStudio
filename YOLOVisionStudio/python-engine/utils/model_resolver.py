import os
import threading
from pathlib import Path

# Directories are provided by the .NET launcher via environment variables.
# Fallbacks point into the ASP.NET wwwroot folder next to this repo.
_default_models = Path(__file__).resolve().parents[2] / "YOLOVisionStudio" / "wwwroot" / "models"

MODELS_DIR = Path(os.environ.get("YOLO_MODELS_DIR") or _default_models)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_download_lock = threading.Lock()


def resolve_model_path(model_name: str) -> str:
    """Return an absolute path for a model file inside MODELS_DIR if it exists,
    otherwise return the bare name (ultralytics will resolve/download it into
    weights_dir, which main.py points at MODELS_DIR)."""
    name = os.path.basename(model_name.strip())
    candidate = MODELS_DIR / name
    if candidate.exists():
        return str(candidate)
    if os.path.isabs(model_name) and os.path.exists(model_name):
        # Trained/exported weights passed by absolute path: use as-is.
        return model_name
    return model_name


def get_models_dir() -> Path:
    return MODELS_DIR


def list_cached_models() -> list[str]:
    """All weight files currently stored in the shared models directory."""
    if MODELS_DIR.exists():
        return sorted(f.name for f in MODELS_DIR.glob("*.pt"))
    return []


def ensure_model(model_name: str) -> dict:
    """Check local storage first; download into MODELS_DIR only if missing.
    Returns a status dict: {model, exists, downloaded, path, size_mb}."""
    name = os.path.basename(model_name.strip())
    target = MODELS_DIR / name

    if target.exists():
        return {
            "model": name,
            "exists": True,
            "downloaded": False,
            "path": str(target),
            "size_mb": round(target.stat().st_size / (1024 * 1024), 2),
        }

    with _download_lock:
        # Double-check after acquiring lock (another thread may have downloaded).
        if not target.exists():
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(target))

    if not target.exists():
        raise FileNotFoundError(f"Could not download model '{name}'")

    return {
        "model": name,
        "exists": True,
        "downloaded": True,
        "path": str(target),
        "size_mb": round(target.stat().st_size / (1024 * 1024), 2),
    }
