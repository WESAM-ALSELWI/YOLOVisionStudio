import os
import glob
import hashlib
import shutil
import zipfile
import pathlib
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from models.schemas import DatasetAnalysisRequest, DatasetAnalysisResponse, AugmentationRequest, AugmentationResponse
from utils.image_utils import numpy_to_base64, base64_to_numpy
from utils.yolo_manager import build_dataset_yaml

router = APIRouter(prefix="/dataset", tags=["Dataset"])

# ---------------------------------------------------------------------------
# Built-in sample datasets (real, downloadable, tiny enough for quick runs)
# ---------------------------------------------------------------------------
_default_datasets = pathlib.Path(__file__).resolve().parents[2] / "YOLOVisionStudio" / "wwwroot" / "datasets"
DATASETS_DIR = pathlib.Path(os.environ.get("YOLO_DATASETS_DIR") or _default_datasets)

SAMPLE_DATASETS = {
    "coco8": {
        "name": "COCO8 Sandbox",
        "description": "The official Ultralytics COCO8 mini-dataset: 8 real COCO images used to verify the full training pipeline end-to-end.",
        "classes_count": 80,
        "images": 8,
        "size_mb": 1,
        "difficulty": "Beginner",
        "task": "detect",
        "suggested_model": "yolo11n.pt",
    },
    "coco128": {
        "name": "COCO128 Starter",
        "description": "128 real COCO training images with labels — big enough to see the model actually learn, small enough for CPU/GPU quick runs.",
        "classes_count": 80,
        "images": 128,
        "size_mb": 6,
        "difficulty": "Intermediate",
        "task": "detect",
        "suggested_model": "yolo11n.pt",
    },
    "dota8": {
        "name": "DOTA8 Aerial (OBB)",
        "description": "8 large aerial drone images from DOTA with oriented bounding-box annotations. Demonstrates rotated object detection.",
        "classes_count": 15,
        "images": 8,
        "size_mb": 1,
        "difficulty": "Advanced",
        "task": "obb",
        "suggested_model": "yolo11n-obb.pt",
    },
}


def _sample_yaml_path(key: str) -> pathlib.Path:
    """Build/return a data.yaml for an installed sample dataset."""
    folder = DATASETS_DIR / key
    yaml_path = folder / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    # Prefer the official yaml shipped inside the ultralytics package.
    try:
        from ultralytics.utils import ROOT as UL_ROOT
        official = UL_ROOT / "cfg" / "datasets" / f"{key}.yml"
        if official.exists():
            import yaml
            with open(official, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            data["path"] = str(folder)
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            return yaml_path
    except Exception:
        pass

    names = SAMPLE_DATASETS.get(key, {})
    nc = int(names.get("classes_count", 1))
    content = (
        f"path: {folder.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {nc}\n"
        "names:\n" + "\n".join(f"  - class_{i}" for i in range(nc)) + "\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


@router.get("/samples")
async def list_samples():
    """List built-in sample datasets and whether each is already installed locally."""
    items = []
    for key, meta in SAMPLE_DATASETS.items():
        installed = (DATASETS_DIR / key / "data.yaml").exists()
        items.append({"key": key, **meta, "installed": installed})
    return {"directory": str(DATASETS_DIR), "datasets": items}


class InstallSampleRequest(BaseModel):
    name: str


_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def _download_release_asset(key: str, dest: pathlib.Path) -> None:
    """Download an official ultralytics sample-dataset zip (with a fallback
    through attempt_download_asset for environments that prefer it)."""
    import urllib.request

    url = f"{_ASSET_BASE}/{key}.zip"
    print(f"[dataset] downloading {url} -> {dest}")
    try:
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
    except Exception as e:
        # Fallback to ultralytics' own downloader before giving up.
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(dest))
        except Exception:
            raise HTTPException(status_code=502, detail=f"Download failed ({url}): {e}")


@router.post("/install-sample")
async def install_sample(request: InstallSampleRequest):
    """Install a sample dataset: reuse the local copy if it exists, otherwise
    download + extract it into the shared datasets directory."""
    key = request.name.strip().lower()
    meta = SAMPLE_DATASETS.get(key)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown sample dataset '{key}'")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    folder = DATASETS_DIR / key
    yaml_path = folder / "data.yaml"

    downloaded = False
    if not yaml_path.exists():
        zip_path = DATASETS_DIR / f"{key}.zip"
        try:
            if not zip_path.exists():
                _download_release_asset(key, zip_path)
                if not zip_path.exists():
                    raise HTTPException(status_code=502, detail=f"Dataset archive '{key}.zip' could not be downloaded")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(DATASETS_DIR)
            downloaded = True
        except FileNotFoundError as e:
            raise HTTPException(status_code=502, detail=f"Dataset archive not available: {e}")
        except zipfile.BadZipFile:
            raise HTTPException(status_code=502, detail="Downloaded dataset archive is corrupted")

    if not yaml_path.exists():
        yaml_path = _sample_yaml_path(key)
    if not folder.exists():
        raise HTTPException(status_code=500, detail="Dataset folder missing after extraction")

    # Count real images for accurate reporting.
    img_count = len(glob.glob(str(folder / "images" / "**" / "*.jpg"), recursive=True))
    return {
        "success": True,
        "key": key,
        "name": meta["name"],
        "downloaded": downloaded,
        "installed": True,
        "yaml_path": str(yaml_path),
        "storage_path": str(folder),
        "images": img_count or meta["images"],
        "classes_count": meta["classes_count"],
        "task": meta["task"],
        "difficulty": meta["difficulty"],
        "message": ("Downloaded and extracted." if downloaded else "Already installed locally — reused existing copy."),
    }

class GenerateYamlRequest(BaseModel):
    train_path: str
    val_path: str
    test_path: str = ""
    class_names: List[str]
    output_path: str

@router.post("/analyze", response_model=DatasetAnalysisResponse)
async def analyze_dataset(request: DatasetAnalysisRequest):
    """Analyze a YOLO-format dataset folder."""
    base_path = request.dataset_path
    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Dataset path not found")
        
    image_paths = glob.glob(os.path.join(base_path, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(base_path, "**", "*.png"), recursive=True)
                  
    total_images = len(image_paths)
    total_labeled = 0
    class_distribution = {}
    missing_annotations = 0
    corrupted_images = 0
    duplicate_images = 0
    
    hashes = set()
    sample_images = []
    
    for i, img_path in enumerate(image_paths):
        # Corrupted check
        img = cv2.imread(img_path)
        if img is None:
            corrupted_images += 1
            continue
            
        if i < 5:
            # Downscale for sample thumbnails
            thumb = cv2.resize(img, (200, 200))
            sample_images.append(numpy_to_base64(thumb))
            
        # Duplicate check
        with open(img_path, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
            if h in hashes:
                duplicate_images += 1
            else:
                hashes.add(h)
                
        # Label check
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(label_path):
            total_labeled += 1
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = parts[0]
                            class_distribution[cls_id] = class_distribution.get(cls_id, 0) + 1
            except:
                pass
        else:
            missing_annotations += 1
            
    return DatasetAnalysisResponse(
        total_images=total_images,
        total_labeled=total_labeled,
        total_unlabeled=total_images - total_labeled,
        classes=list(class_distribution.keys()),
        class_distribution=class_distribution,
        missing_annotations=missing_annotations,
        duplicate_images=duplicate_images,
        corrupted_images=corrupted_images,
        sample_images=sample_images
    )

@router.post("/augment", response_model=AugmentationResponse)
async def augment_image(request: AugmentationRequest):
    """Apply basic augmentations using cv2."""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    img = cv2.imread(request.image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")
        
    orig_b64 = numpy_to_base64(img)
    aug_b64_list = []
    
    for _ in range(request.count):
        aug_img = img.copy()
        params = request.augmentations
        
        # Simple cv2 augmentations
        if params.get('horizontal_flip'):
            aug_img = cv2.flip(aug_img, 1)
        if params.get('vertical_flip'):
            aug_img = cv2.flip(aug_img, 0)
        if params.get('blur'):
            kernel = params['blur']
            aug_img = cv2.blur(aug_img, (kernel, kernel))
            
        aug_b64_list.append(numpy_to_base64(aug_img))
        
    return AugmentationResponse(
        original_image_base64=orig_b64,
        augmented_images=aug_b64_list
    )

@router.post("/generate-yaml")
async def generate_yaml(request: GenerateYamlRequest):
    """Generate data.yaml for dataset."""
    success = build_dataset_yaml(
        train_path=request.train_path,
        val_path=request.val_path,
        test_path=request.test_path,
        class_names=request.class_names,
        output_path=request.output_path
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write YAML file")
    return {"success": True, "path": request.output_path}


# ---------------------------------------------------------------------------
# Dataset splitting: build a proper YOLO structure (images/{train,val,test} +
# labels/{train,val,test} + data.yaml) from any folder of images.
# ---------------------------------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class SplitRequest(BaseModel):
    dataset_path: str
    train_pct: int = 70
    val_pct: int = 15
    test_pct: int = 15
    seed: int = 42


def _find_split_candidates(root: str) -> list:
    """All images in the dataset that are NOT already inside images/train|val|test."""
    cands = []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace("\\", "/").split("/")
        if len(rel) >= 2 and rel[0] == "images" and rel[1] in ("train", "val", "test"):
            continue
        for f in filenames:
            if f.lower().endswith(IMG_EXTS):
                cands.append(os.path.join(dirpath, f))
    return sorted(cands)


def _find_label_for(img_path: str, root: str) -> str | None:
    """YOLO label file: parallel labels/ tree, or a sibling .txt."""
    rel = os.path.relpath(img_path, root)
    base, _ = os.path.splitext(rel)
    # Handle common layouts where images live under an images/ subtree:
    # images/train2017/a.jpg -> labels/train2017/a.txt
    parts = base.replace("\\", "/").split("/")
    mapped = None
    if "images" in parts:
        i = parts.index("images")
        mapped_parts = parts[:i] + ["labels"] + parts[i + 1:]
        mapped = "/".join(mapped_parts)
    candidates = [
        os.path.join(root, "labels", base + ".txt"),
    ]
    if mapped:
        candidates.append(os.path.join(root, mapped.replace("/", os.sep) + ".txt"))
    candidates.append(os.path.splitext(img_path)[0] + ".txt")
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _scan_max_class_id(label_files: list) -> int:
    """Highest class id found in the labels (for nc in data.yaml)."""
    max_cls = -1
    for lf in label_files:
        if not lf:
            continue
        try:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        max_cls = max(max_cls, int(float(parts[0])))
        except Exception:
            continue
    return max_cls


@router.post("/split")
async def split_dataset(request: SplitRequest):
    """Split any folder of images into the YOLO train/val/test structure and
    write a data.yaml. Files are COPIED (originals are never touched)."""
    import random
    import shutil
    import yaml as pyyaml

    root = request.dataset_path
    if not os.path.isdir(root):
        raise HTTPException(status_code=404, detail=f"Dataset path not found: {root}")
    if request.train_pct + request.val_pct + request.test_pct != 100:
        raise HTTPException(status_code=400, detail="train + val + test must equal 100")
    if request.train_pct <= 0:
        raise HTTPException(status_code=400, detail="train percentage must be > 0")

    candidates = _find_split_candidates(root)
    if not candidates:
        raise HTTPException(status_code=400, detail="No images found to split in this folder")

    rng = random.Random(request.seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * request.train_pct / 100)
    n_val = round(n * request.val_pct / 100)
    n_test = n - n_train - n_val  # remainder keeps the sum exact

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    copied, labels_found = 0, 0
    label_files_used = []
    for split_name, files in splits.items():
        img_dir = os.path.join(root, "images", split_name)
        lbl_dir = os.path.join(root, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for src in files:
            dst = os.path.join(img_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied += 1
            lbl = _find_label_for(src, root)
            if lbl:
                shutil.copy2(lbl, os.path.join(lbl_dir, os.path.basename(lbl)))
                label_files_used.append(lbl)
                labels_found += 1

    max_cls = _scan_max_class_id(label_files_used)
    nc = max_cls + 1 if max_cls >= 0 else 1
    names = [f"class_{i}" for i in range(nc)]

    yaml_data = {
        "path": os.path.abspath(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        pyyaml.safe_dump(yaml_data, f, default_flow_style=False)

    return {
        "success": True,
        "total_images": n,
        "copied": copied,
        "labels_found": labels_found,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "yaml_path": yaml_path,
        "classes": nc,
        "message": f"Split {n} images into train/val/test ({n_train}/{n_val}/{n_test}). {labels_found} label files matched.",
    }


@router.get("/split-status")
async def split_status(path: str):
    """Check whether a dataset folder already follows the YOLO structure."""
    if not os.path.isdir(path):
        return {"exists": False, "is_yolo_structured": False, "images": 0}
    train_dir = os.path.join(path, "images", "train")
    has_yaml = os.path.exists(os.path.join(path, "data.yaml"))
    train_imgs = len(glob.glob(os.path.join(train_dir, "**", "*"), recursive=True)) if os.path.isdir(train_dir) else 0
    train_imgs = len([f for f in glob.glob(os.path.join(train_dir, "*")) if f.lower().endswith(IMG_EXTS)]) if os.path.isdir(train_dir) else 0
    unsplit = len(_find_split_candidates(path))
    return {
        "exists": True,
        "is_yolo_structured": has_yaml and train_imgs > 0,
        "train_images": train_imgs,
        "unsplit_images": unsplit,
    }
