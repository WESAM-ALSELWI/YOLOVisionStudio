import uuid
import threading
import datetime
import os
from typing import Dict
from fastapi import APIRouter, HTTPException

from models.schemas import TrainingRequest, TrainingStartResponse, TrainingMetricsResponse, StopTrainingResponse
from utils.yolo_manager import parse_results_csv
from utils.model_resolver import resolve_model_path
from utils.training_viz import TrainingVisualizer
from ultralytics import YOLO

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

router = APIRouter(prefix="/training", tags=["Training"])

class JobState:
    def __init__(self, job_id: str, request: TrainingRequest):
        self.job_id = job_id
        self.status = "pending"
        self.request = request
        self.thread: threading.Thread = None
        self.results_dir = os.path.join(ENGINE_DIR, "runs", "train", request.name)
        self.error_message = ""
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self._stop_event = threading.Event()
        # Real-time visualization state fed by ultralytics callbacks
        self.visualizer: TrainingVisualizer = TrainingVisualizer()

jobs_store: Dict[str, JobState] = {}

def run_training_task(job_id: str, request: TrainingRequest, job_state: JobState):
    job_state.status = "running"
    try:
        model_path = resolve_model_path(request.model)
        model = YOLO(model_path)

        augment = request.augment_params or {}
        job_state.visualizer.attach(model)

        def _on_train_start(trainer):
            # Ultralytics resolves the actual output directory itself —
            # remember it so metric parsing reads the right results.csv.
            try:
                job_state.results_dir = str(trainer.save_dir)
            except Exception:
                pass

        model.add_callback("on_train_start", _on_train_start)
        train_kwargs = dict(
            data=request.dataset_yaml_path,
            epochs=request.epochs,
            imgsz=request.imgsz,
            batch=request.batch,
            optimizer=request.optimizer,
            lr0=request.lr0,
            weight_decay=request.weight_decay,
            momentum=request.momentum,
            warmup_epochs=request.warmup_epochs,
            freeze=request.freeze,
            project=os.path.join(ENGINE_DIR, "runs", "train"),
            name=request.name,
            device=request.device or None,
            # Windows DataLoader workers rely on shared memory mappings that
            # often fail (WinError 1455 / bad allocation) — load in-process there.
            workers=request.workers if request.workers is not None else (0 if os.name == "nt" else 8),
            exist_ok=True,
        )
        if "mosaic" in augment:
            train_kwargs["mosaic"] = float(augment["mosaic"])
        if "mixup" in augment:
            train_kwargs["mixup"] = float(augment["mixup"])
        model.train(**train_kwargs)

        if job_state._stop_event.is_set():
            job_state.status = "cancelled"
        else:
            job_state.status = "completed"

    except Exception as e:
        job_state.status = "failed"
        job_state.error_message = str(e)
    finally:
        job_state.end_time = datetime.datetime.now()
        try:
            model.clear_callback("on_train_start")
        except Exception:
            pass
        with job_state.visualizer.lock:
            if job_state.status == "failed":
                job_state.visualizer.status = "failed"
                job_state.visualizer.error_message = job_state.error_message
            elif job_state.visualizer.status != "completed":
                job_state.visualizer.status = job_state.status

@router.post("/start", response_model=TrainingStartResponse)
async def start_training(request: TrainingRequest):
    job_id = str(uuid.uuid4())
    job_state = JobState(job_id, request)
    
    t = threading.Thread(target=run_training_task, args=(job_id, request, job_state))
    t.daemon = True
    job_state.thread = t
    jobs_store[job_id] = job_state
    
    t.start()
    
    return TrainingStartResponse(
        job_id=job_id,
        status="started",
        message="Training started in background."
    )

@router.get("/{job_id}/live-layer/{layer_index}")
async def live_layer(job_id: str, layer_index: int):
    """Deep inspection of one layer's latest REAL activation — used when the
    user clicks a layer in the live network flow."""
    state = jobs_store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    vis = getattr(state, "visualizer", None)
    if vis is None:
        raise HTTPException(status_code=404, detail="Job has no visualizer")
    from utils.training_viz import layer_detail
    return layer_detail(vis, layer_index)


@router.get("/{job_id}/metrics", response_model=TrainingMetricsResponse)
async def get_training_metrics(job_id: str):
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]
    vis = getattr(job, "visualizer", None)

    csv_path = os.path.join(job.results_dir, "results.csv")
    metrics_list = parse_results_csv(csv_path)

    latest_metrics = {}
    if metrics_list:
        latest_metrics = metrics_list[-1]
    elif vis is not None:
        # Fall back to the live callback metrics (real values from trainer).
        with vis.lock:
            latest_metrics = {k: v for k, v in vis.metrics.items()}

    # Helper to safely extract metrics
    def get_metric(name: str, default: float = 0.0) -> float:
        for k, v in latest_metrics.items():
            if name.lower() in k.lower():
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return default

    epoch = int(get_metric('epoch', 0))
    status = job.status
    error_message = job.error_message

    return TrainingMetricsResponse(
        job_id=job_id,
        epoch=epoch,
        total_epochs=job.request.epochs,
        box_loss=get_metric('box_loss'),
        cls_loss=get_metric('cls_loss'),
        dfl_loss=get_metric('dfl_loss'),
        precision=get_metric('metrics/mAP50(B)', get_metric('precision')),
        recall=get_metric('recall'),
        map50=get_metric('mAP50'),
        map5095=get_metric('mAP50-95'),
        lr=get_metric('lr'),
        status=status,
        is_finished=status in ["completed", "failed", "cancelled"],
        is_failed=status == "failed",
        error_message=error_message
    )

@router.post("/{job_id}/stop", response_model=StopTrainingResponse)
async def stop_training(job_id: str):
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs_store[job_id]
    if job.status == "running":
        job._stop_event.set()
        job.status = "cancelled"
        # Note: Killing threads in python is inherently unsafe. 
        # Ultralytics doesn't expose a clean stop flag during model.train() easily from outside.
        # A more robust system would run it in a subprocess and terminate the process.
        # For this requirement, setting status is provided.
        return StopTrainingResponse(success=True, message="Cancellation signal sent.")
    
    return StopTrainingResponse(success=False, message=f"Job is not running (status: {job.status}).")

@router.get("/jobs")
async def list_jobs():
    return [
        {
            "job_id": jid, 
            "status": state.status,
            "model": state.request.model,
            "project": state.request.project_id
        } 
        for jid, state in jobs_store.items()
    ]
