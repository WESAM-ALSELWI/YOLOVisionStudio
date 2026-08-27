import time
import os
import glob
from fastapi import APIRouter, HTTPException
from ultralytics import YOLO
from models.schemas import InferenceRequest, InferenceResponse, DetectionResult
from utils.image_utils import base64_to_numpy, numpy_to_base64, draw_detections
from utils.model_resolver import resolve_model_path, list_cached_models

router = APIRouter(prefix="/inference", tags=["Inference"])

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@router.post("/image", response_model=InferenceResponse)
async def infer_image(request: InferenceRequest):
    """Run YOLO prediction on an image."""
    try:
        model_path = resolve_model_path(request.model_path)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
            
        model = YOLO(model_path)
        
        # Load image
        if request.image_base64:
            img = base64_to_numpy(request.image_base64)
        elif request.image_path and os.path.exists(request.image_path):
            import cv2
            img = cv2.imread(request.image_path)
            if img is None:
                raise ValueError("Could not read image from path")
        else:
            raise HTTPException(status_code=400, detail="Provide image_base64 or valid image_path")
            
        # Inference
        start_time = time.time()
        results = model.predict(
            source=img,
            conf=request.confidence,
            iou=request.iou,
            task=request.task,
            verbose=False
        )
        process_time_ms = (time.time() - start_time) * 1000
        
        result = results[0]
        names = result.names
        
        detections = []
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append(DetectionResult(
                    class_id=cls_id,
                    class_name=names[cls_id],
                    confidence=conf,
                    bbox=[x1, y1, x2, y2]
                ))
                
        # Draw on image
        detections_dict_list = [d.model_dump() for d in detections]
        annotated_img = draw_detections(img, detections_dict_list, names)
        
        return InferenceResponse(
            detections=detections,
            image_base64=numpy_to_base64(annotated_img),
            processing_time_ms=process_time_ms,
            class_names=names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List available model files from runs directory and cached models."""
    models = glob.glob(os.path.join(ENGINE_DIR, "runs", "**", "*.pt"), recursive=True)
    cached = list_cached_models()
    return {"models": models, "cached_models": cached}
