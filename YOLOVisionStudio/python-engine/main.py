import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routers import training, inference, dataset, visualization, system, export, models as models_router, ws_training

app = FastAPI(
    title="YOLO Vision Studio AI Engine",
    description="FastAPI microservice for YOLO Vision Studio",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Include Routers
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(dataset.router)
app.include_router(visualization.router)
app.include_router(system.router)
app.include_router(export.router)
app.include_router(models_router.router)
app.include_router(ws_training.router)

@app.on_event("startup")
async def startup_event():
    print("YOLO Vision Studio AI Engine starting up...")
    # Point ultralytics at the shared ASP.NET wwwroot folders so pretrained
    # weights and datasets are downloaded once and reused from there.
    try:
        from utils.model_resolver import MODELS_DIR
        from routers.dataset import DATASETS_DIR
        from ultralytics import settings
        settings.update({"weights_dir": str(MODELS_DIR), "datasets_dir": str(DATASETS_DIR)})
        print(f"  models dir   : {MODELS_DIR}")
        print(f"  datasets dir : {DATASETS_DIR}")
    except Exception as e:
        print(f"  (could not configure ultralytics dirs: {e})")

@app.on_event("shutdown")
async def shutdown_event():
    print("YOLO Vision Studio AI Engine shutting down...")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "YOLO Vision Studio AI Engine is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
