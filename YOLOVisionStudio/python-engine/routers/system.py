from fastapi import APIRouter
from models.schemas import SystemStatsResponse
from utils.system_monitor import SystemMonitor

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Return current system stats (CPU, RAM, Disk, GPU)."""
    stats = SystemMonitor.get_all_stats()
    return SystemStatsResponse(**stats)

@router.get("/health")
async def system_health():
    """Detailed health check for the system."""
    return {
        "status": "healthy",
        "gpu_available": SystemMonitor.get_gpu_stats()["name"] != "N/A (CPU Only / No NVML)"
    }

@router.get("/gpu-info")
async def get_gpu_info():
    """Detailed GPU information."""
    return SystemMonitor.get_gpu_stats()
