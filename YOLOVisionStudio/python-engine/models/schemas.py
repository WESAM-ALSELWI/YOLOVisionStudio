from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# =======================
# Training Schemas
# =======================
class TrainingRequest(BaseModel):
    project_id: str
    dataset_yaml_path: str
    model: str = "yolo11n.pt"
    task: str = "detect"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: float = 3.0
    freeze: Optional[int] = None
    augment_params: Optional[Dict[str, Any]] = None
    name: str = "exp"
    device: Optional[str] = "cpu"
    workers: Optional[int] = None

class TrainingStartResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingMetricsResponse(BaseModel):
    job_id: str
    epoch: int = 0
    total_epochs: int = 0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map5095: float = 0.0
    lr: float = 0.0
    status: str
    is_finished: bool
    is_failed: bool
    error_message: Optional[str] = None

class StopTrainingResponse(BaseModel):
    success: bool
    message: str

# =======================
# Inference Schemas
# =======================
class InferenceRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    confidence: float = 0.25
    iou: float = 0.45
    task: str = "detect"

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    mask_base64: Optional[str] = None
    keypoints: Optional[List[List[float]]] = None

class InferenceResponse(BaseModel):
    detections: List[DetectionResult]
    image_base64: Optional[str] = None
    processing_time_ms: float
    class_names: Dict[int, str]

# =======================
# Dataset Schemas
# =======================
class DatasetAnalysisRequest(BaseModel):
    dataset_path: str

class DatasetAnalysisResponse(BaseModel):
    total_images: int
    total_labeled: int
    total_unlabeled: int
    classes: List[str]
    class_distribution: Dict[str, int]
    missing_annotations: int
    duplicate_images: int
    corrupted_images: int
    sample_images: List[str]

class AugmentationRequest(BaseModel):
    image_path: str
    augmentations: Dict[str, Any]
    count: int = 1

class AugmentationResponse(BaseModel):
    original_image_base64: str
    augmented_images: List[str]

# =======================
# System Schemas
# =======================
class SystemStatsResponse(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_name: str
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_percent: float
    gpu_temperature: float
    disk_used_gb: float
    disk_total_gb: float

# =======================
# Visualization Schemas
# =======================
class LayerInfo(BaseModel):
    id: str
    name: str
    type: str
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    parameters: int = 0
    flops: float = 0.0
    activation: Optional[str] = None
    kernel: Optional[str] = None
    stride: Optional[str] = None
    details: Optional[str] = None
    plain: Optional[str] = None
    children: List["LayerInfo"] = []

LayerInfo.model_rebuild()

class ArchitectureRequest(BaseModel):
    yolo_version: str
    model_size: str

class ArchitectureResponse(BaseModel):
    layers: List[LayerInfo]
    total_parameters: int
    total_flops: float

class FeatureMapsRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    layer_indices: List[int]

class FeatureMap(BaseModel):
    layer_index: int
    layer_name: str
    shape: List[int]
    channels_base64: List[str]

class FeatureMapsResponse(BaseModel):
    original_image_base64: str
    feature_maps: List[FeatureMap]

# =======================
# Export Schemas
# =======================
class ExportRequest(BaseModel):
    model_path: str
    format: str
    imgsz: int = 640
    half: bool = False
    dynamic: bool = False

class ExportResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    file_size_mb: float = 0.0
    message: str

# =======================
# Real-time Training Visualization Schemas
# =======================
class LayerStatistics(BaseModel):
    layer_index: int
    layer_name: str
    layer_type: str
    mean: float
    std: float
    min: float
    max: float
    sparsity: float  # Percentage of near-zero activations
    dead_neurons: int  # Count of neurons with zero gradient
    grad_norm: float  # L2 norm of gradients

class WeightVisualization(BaseModel):
    layer_index: int
    layer_name: str
    weight_shape: List[int]
    weights_base64: str  # Grid visualization of filters
    weight_stats: Dict[str, float]
    bias_base64: Optional[str] = None

class GradCAMRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    target_layer: str
    class_id: Optional[int] = None

class GradCAMResponse(BaseModel):
    original_image_base64: str
    heatmap_base64: str
    overlay_base64: str
    target_class: Optional[str] = None
    confidence: Optional[float] = None

class AttentionMapRequest(BaseModel):
    model_path: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None

class AttentionMapResponse(BaseModel):
    original_image_base64: str
    attention_maps: List[Dict[str, Any]]  # Multi-scale attention maps
    overlay_base64: str

class TrainingVisualizationSnapshot(BaseModel):
    job_id: str
    epoch: int
    batch: int
    layer_statistics: List[LayerStatistics]
    weight_visualizations: List[WeightVisualization]
    grad_norms: Dict[int, float]
    activation_heatmaps: List[Dict[str, Any]]
