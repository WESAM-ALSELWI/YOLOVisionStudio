from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.model_resolver import ensure_model, resolve_model_path, list_cached_models, MODELS_DIR

router = APIRouter(prefix="/models", tags=["Models"])


class EnsureModelRequest(BaseModel):
    model: str


@router.get("")
async def get_models():
    """List pretrained weights available in the shared wwwroot/models directory."""
    models = []
    for name in list_cached_models():
        p = MODELS_DIR / name
        models.append({
            "name": name,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "path": str(p),
        })
    return {"directory": str(MODELS_DIR), "models": models}


@router.get("/status/{name}")
async def model_status(name: str):
    """Check whether a model exists locally (never downloads)."""
    path = MODELS_DIR / name
    return {
        "model": name,
        "exists": path.exists(),
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0,
    }


@router.post("/ensure")
async def ensure(request: EnsureModelRequest):
    """Download the model into the shared models directory only if it is not
    already cached there. Safe to call repeatedly."""
    try:
        info = ensure_model(request.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model download failed: {e}")
    return info
