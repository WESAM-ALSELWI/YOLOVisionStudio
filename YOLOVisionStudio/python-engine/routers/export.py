import os
from fastapi import APIRouter, HTTPException
from models.schemas import ExportRequest, ExportResponse
from ultralytics import YOLO
from utils.model_resolver import resolve_model_path

router = APIRouter(prefix="/export", tags=["Export"])

@router.post("/model", response_model=ExportResponse)
async def export_model(request: ExportRequest):
    """Export YOLO model to specified format."""
    model_path = resolve_model_path(request.model_path)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model path not found: {request.model_path}")

    try:
        model = YOLO(model_path)
        
        # Format mapping just to ensure valid inputs
        valid_formats = ['onnx', 'torchscript', 'openvino', 'coreml', 'engine', 'pb', 'tflite', 'edgetpu', 'tfjs']
        if request.format.lower() not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
            
        exported_path = model.export(
            format=request.format.lower(),
            imgsz=request.imgsz,
            half=request.half,
            dynamic=request.dynamic
        )
        
        file_size_mb = 0.0
        if isinstance(exported_path, str) and os.path.exists(exported_path):
            file_size_mb = os.path.getsize(exported_path) / (1024 * 1024)
            
        return ExportResponse(
            success=True,
            output_path=str(exported_path),
            file_size_mb=round(file_size_mb, 2),
            message=f"Successfully exported to {request.format}"
        )
    except Exception as e:
        return ExportResponse(
            success=False,
            message=str(e)
        )

@router.get("/formats")
async def get_export_formats():
    """Return list of supported export formats."""
    return {
        "formats": [
            {"id": "onnx", "name": "ONNX", "description": "Open Neural Network Exchange"},
            {"id": "torchscript", "name": "TorchScript", "description": "PyTorch TorchScript"},
            {"id": "openvino", "name": "OpenVINO", "description": "Intel OpenVINO"},
            {"id": "coreml", "name": "CoreML", "description": "Apple CoreML"}
        ]
    }
