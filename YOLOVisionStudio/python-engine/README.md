# YOLO Vision Studio AI Engine

This is the Python FastAPI microservice for YOLO Vision Studio, powering all AI-related operations including model training, inference, dataset management, and visualization.

## Prerequisites

- Python 3.10+
- Conda environment `AmalAI` containing:
  - `ultralytics`
  - `opencv-python`
  - `torch` & `torchvision`
  - `tensorboard`
  - `onnxruntime`

## Installation & Startup

1. Ensure the AmalAI environment is created.
2. Install the required FastAPI packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the engine:
   ```bash
   start.bat
   ```
   Or manually:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Architecture

- **`main.py`**: The FastAPI application entrypoint.
- **`models/`**: Pydantic schemas for request/response validation.
- **`routers/`**: Grouped endpoints for `training`, `inference`, `dataset`, `visualization`, `system`, and `export`.
- **`utils/`**: Shared utilities for system monitoring, image processing, and YOLO interactions.

## Key Endpoints

- `GET /health` : Basic health check.
- `POST /training/start` : Start a YOLO training job.
- `POST /inference/image` : Run YOLO object detection.
- `GET /system/stats` : Retrieve system CPU/GPU/RAM metrics.
- `POST /visualization/feature-maps` : Extract layer feature maps.
