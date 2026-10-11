using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.ViewModels;


namespace YOLOVisionStudio.Services
{
    /// <summary>Payload for the Python engine /training/start endpoint (snake_case JSON).</summary>
    public class TrainingRequest
    {
        [JsonPropertyName("project_id")] public string ProjectId { get; set; } = "1";
        [JsonPropertyName("dataset_yaml_path")] public string DatasetYamlPath { get; set; } = string.Empty;
        [JsonPropertyName("model")] public string Model { get; set; } = "yolo11n.pt";
        [JsonPropertyName("task")] public string Task { get; set; } = "detect";
        [JsonPropertyName("epochs")] public int Epochs { get; set; } = 100;
        [JsonPropertyName("imgsz")] public int Imgsz { get; set; } = 640;
        [JsonPropertyName("batch")] public int Batch { get; set; } = 16;
        [JsonPropertyName("optimizer")] public string Optimizer { get; set; } = "auto";
        [JsonPropertyName("lr0")] public double Lr0 { get; set; } = 0.01;
        [JsonPropertyName("lrf")] public double Lrf { get; set; } = 0.01;
        [JsonPropertyName("weight_decay")] public double WeightDecay { get; set; } = 0.0005;
        [JsonPropertyName("momentum")] public double Momentum { get; set; } = 0.937;
        [JsonPropertyName("warmup_epochs")] public double WarmupEpochs { get; set; } = 3.0;
        [JsonPropertyName("freeze")] public int? Freeze { get; set; }
        [JsonPropertyName("augment_params")] public Dictionary<string, object>? AugmentParams { get; set; }
        [JsonPropertyName("name")] public string Name { get; set; } = "exp";
        [JsonPropertyName("device")] public string? Device { get; set; }

        /// <summary>Local (.NET) job id — ignored by the Python engine.</summary>
        public int JobId { get; set; }
    }

    public class TrainingStartResponse
    {
        [JsonPropertyName("job_id")] public string PythonJobId { get; set; } = string.Empty;
        [JsonPropertyName("status")] public string Status { get; set; } = string.Empty;
        [JsonPropertyName("message")] public string Message { get; set; } = string.Empty;
        public bool Success { get; set; } = true;
    }

    public class TrainingMetricsResponse
    {
        [JsonPropertyName("epoch")] public int Epoch { get; set; }
        [JsonPropertyName("total_epochs")] public int TotalEpochs { get; set; }
        [JsonPropertyName("box_loss")] public double BoxLoss { get; set; }
        [JsonPropertyName("cls_loss")] public double ClsLoss { get; set; }
        [JsonPropertyName("dfl_loss")] public double DflLoss { get; set; }
        [JsonPropertyName("precision")] public double Precision { get; set; }
        [JsonPropertyName("recall")] public double Recall { get; set; }
        [JsonPropertyName("map50")] public double mAP50 { get; set; }
        [JsonPropertyName("map5095")] public double mAP5095 { get; set; }
        [JsonPropertyName("lr")] public double LearningRate { get; set; }
        [JsonPropertyName("status")] public string Status { get; set; } = string.Empty;
        [JsonPropertyName("is_finished")] public bool IsFinished { get; set; }
        [JsonPropertyName("is_failed")] public bool IsFailed { get; set; }
        [JsonPropertyName("error_message")] public string? ErrorMessage { get; set; }
    }

    public class StopTrainingResponse { public bool Success { get; set; } }
    public class InferenceRequest { public string ImagePath { get; set; } = string.Empty; }
    public class InferenceResponse { public string JsonResults { get; set; } = string.Empty; }
    public class DatasetAnalysisResponse { public string StatsJson { get; set; } = string.Empty; }
    public class SystemStatsResponse { public double GpuUsage { get; set; } public string GpuName { get; set; } = string.Empty; }
    public class FeatureMapsRequest
    {
        public string ImagePath { get; set; } = string.Empty;
        public string? ModelPath { get; set; }
        public List<int> LayerIndices { get; set; } = new List<int> { 0, 2, 4, 8 };
    }
    public class FeatureMapsResponse
    {
        public string Base64Image { get; set; } = string.Empty;
        public List<FeatureLayerData> FeatureLayers { get; set; } = new();
    }
    public class FeatureLayerData
    {
        public int LayerIndex { get; set; }
        public string LayerName { get; set; } = string.Empty;
        public List<int> Shape { get; set; } = new();
        public List<string> ChannelsBase64 { get; set; } = new();
    }

    public class ExportRequest { public int ModelId { get; set; } public string Format { get; set; } = string.Empty; }
    public class ExportResponse { public string DownloadUrl { get; set; } = string.Empty; public bool Success { get; set; } }
    public class LayerNode
    {
        [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
        [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
        [JsonPropertyName("type")] public string Type { get; set; } = string.Empty;
        [JsonPropertyName("input_shape")] public List<int>? InputShape { get; set; }
        [JsonPropertyName("output_shape")] public List<int>? OutputShape { get; set; }
        [JsonPropertyName("parameters")] public long Parameters { get; set; }
        [JsonPropertyName("flops")] public double Flops { get; set; }
        [JsonPropertyName("activation")] public string? Activation { get; set; }
        [JsonPropertyName("kernel")] public string? Kernel { get; set; }
        [JsonPropertyName("stride")] public string? Stride { get; set; }
        [JsonPropertyName("details")] public string? Details { get; set; }
        [JsonPropertyName("plain")] public string? Plain { get; set; }
        [JsonPropertyName("children")] public List<LayerNode> Children { get; set; } = new();
    }

    public class ArchitectureResponse
    {
        [JsonPropertyName("layers")] public List<LayerNode> Layers { get; set; } = new();
        [JsonPropertyName("total_parameters")] public long TotalParameters { get; set; }
        [JsonPropertyName("total_flops")] public double TotalFlops { get; set; }
    }
    public class AugmentationRequest { public string ImagePath { get; set; } = string.Empty; }
    public class AugmentationPreviewResponse { public string Base64Image { get; set; } = string.Empty; }

    // ----- Model management (wwwroot/models) -----
    public class ModelStatusResponse
    {
        [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
        [JsonPropertyName("exists")] public bool Exists { get; set; }
        [JsonPropertyName("path")] public string Path { get; set; } = string.Empty;
        [JsonPropertyName("size_mb")] public double SizeMb { get; set; }
    }

    public class EnsureModelResponse : ModelStatusResponse
    {
        [JsonPropertyName("downloaded")] public bool Downloaded { get; set; }
    }

    // ----- Sample datasets (wwwroot/datasets) -----
    public class InstallSampleRequest
    {
        [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    }

    // ----- Dataset splitting (build YOLO train/val/test structure) -----
    public class SplitDatasetRequest
    {
        [JsonPropertyName("dataset_path")] public string DatasetPath { get; set; } = string.Empty;
        [JsonPropertyName("train_pct")] public int TrainPct { get; set; } = 70;
        [JsonPropertyName("val_pct")] public int ValPct { get; set; } = 15;
        [JsonPropertyName("test_pct")] public int TestPct { get; set; } = 15;
    }

    public class SplitDatasetResponse
    {
        [JsonPropertyName("success")] public bool Success { get; set; }
        [JsonPropertyName("total_images")] public int TotalImages { get; set; }
        [JsonPropertyName("labels_found")] public int LabelsFound { get; set; }
        [JsonPropertyName("counts")] public Dictionary<string, int> Counts { get; set; } = new();
        [JsonPropertyName("yaml_path")] public string YamlPath { get; set; } = string.Empty;
        [JsonPropertyName("classes")] public int Classes { get; set; }
        [JsonPropertyName("message")] public string Message { get; set; } = string.Empty;
    }

    public class SplitStatusResponse
    {
        [JsonPropertyName("exists")] public bool Exists { get; set; }
        [JsonPropertyName("is_yolo_structured")] public bool IsYoloStructured { get; set; }
        [JsonPropertyName("train_images")] public int TrainImages { get; set; }
        [JsonPropertyName("unsplit_images")] public int UnsplitImages { get; set; }
    }

    public class InstallSampleResponse
    {
        [JsonPropertyName("success")] public bool Success { get; set; }
        [JsonPropertyName("key")] public string Key { get; set; } = string.Empty;
        [JsonPropertyName("name")] public string DatasetName { get; set; } = string.Empty;
        [JsonPropertyName("downloaded")] public bool Downloaded { get; set; }
        [JsonPropertyName("installed")] public bool Installed { get; set; }
        [JsonPropertyName("yaml_path")] public string YamlPath { get; set; } = string.Empty;
        [JsonPropertyName("storage_path")] public string StoragePath { get; set; } = string.Empty;
        [JsonPropertyName("images")] public int Images { get; set; }
        [JsonPropertyName("classes_count")] public int ClassesCount { get; set; }
        [JsonPropertyName("task")] public string Task { get; set; } = "detect";
        [JsonPropertyName("difficulty")] public string Difficulty { get; set; } = string.Empty;
        [JsonPropertyName("message")] public string Message { get; set; } = string.Empty;
    }

    public interface IYoloAiService
    {
        Task<TrainingStartResponse> StartTraining(TrainingRequest request);
        Task<TrainingMetricsResponse> GetTrainingMetrics(string jobId);
        Task<StopTrainingResponse> StopTraining(string jobId);
        Task<ModelStatusResponse> GetModelStatus(string modelName);
        Task<EnsureModelResponse> EnsureModel(string modelName);
        Task<List<CachedModelInfo>> ListCachedModels();
        Task<InstallSampleResponse> InstallSampleDataset(string key);
        Task<SplitDatasetResponse> SplitDataset(SplitDatasetRequest request);
        Task<SplitStatusResponse> GetSplitStatus(string datasetPath);
        Task<InferenceResponse> RunInference(InferenceRequest request);
        Task<DatasetAnalysisResponse> AnalyzeDataset(string datasetPath);
        Task<SystemStatsResponse> GetSystemStats();
        Task<FeatureMapsResponse> GetFeatureMaps(FeatureMapsRequest request);
        Task<ExportResponse> ExportModel(ExportRequest request);
        Task<ArchitectureResponse> GetModelArchitecture(string yoloVersion, string modelSize);
        Task<AugmentationPreviewResponse> GetAugmentationPreview(AugmentationRequest request);
        Task<GradCAMResponse> GenerateGradCAM(GradCAMRequest request);
        Task<AttentionMapResponse> GenerateAttentionMaps(AttentionMapRequest request);
        Task<WeightVisualization> VisualizeWeights(string modelPath, int layerIndex);
    }

    public class CachedModelInfo
    {
        [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
        [JsonPropertyName("size_mb")] public double SizeMb { get; set; }
        [JsonPropertyName("path")] public string Path { get; set; } = string.Empty;
    }

    // ----- Real XAI Visualization -----
    public class GradCAMRequest
    {
        [JsonPropertyName("model_path")] public string ModelPath { get; set; } = string.Empty;
        [JsonPropertyName("image_path")] public string? ImagePath { get; set; }
        [JsonPropertyName("image_base64")] public string? ImageBase64 { get; set; }
        [JsonPropertyName("target_layer")] public string TargetLayer { get; set; } = "model.22";
        [JsonPropertyName("class_id")] public int? ClassId { get; set; }
    }

    public class GradCAMResponse
    {
        [JsonPropertyName("original_image_base64")] public string OriginalImageBase64 { get; set; } = string.Empty;
        [JsonPropertyName("heatmap_base64")] public string HeatmapBase64 { get; set; } = string.Empty;
        [JsonPropertyName("overlay_base64")] public string OverlayBase64 { get; set; } = string.Empty;
        [JsonPropertyName("target_class")] public string? TargetClass { get; set; }
        [JsonPropertyName("confidence")] public double? Confidence { get; set; }
    }

    public class AttentionMapRequest
    {
        [JsonPropertyName("model_path")] public string ModelPath { get; set; } = string.Empty;
        [JsonPropertyName("image_path")] public string? ImagePath { get; set; }
        [JsonPropertyName("image_base64")] public string? ImageBase64 { get; set; }
    }

    public class AttentionMapResponse
    {
        [JsonPropertyName("original_image_base64")] public string OriginalImageBase64 { get; set; } = string.Empty;
        [JsonPropertyName("attention_maps")] public List<Dictionary<string, object>> AttentionMaps { get; set; } = new();
        [JsonPropertyName("overlay_base64")] public string OverlayBase64 { get; set; } = string.Empty;
    }

    public class WeightVisualization
    {
        [JsonPropertyName("layer_index")] public int LayerIndex { get; set; }
        [JsonPropertyName("layer_name")] public string LayerName { get; set; } = string.Empty;
        [JsonPropertyName("weight_shape")] public List<int> WeightShape { get; set; } = new();
        [JsonPropertyName("weights_base64")] public string WeightsBase64 { get; set; } = string.Empty;
        [JsonPropertyName("weight_stats")] public Dictionary<string, double> WeightStats { get; set; } = new();
        [JsonPropertyName("bias_base64")] public string? BiasBase64 { get; set; }
    }

    public class LayerStatistics
    {
        [JsonPropertyName("layer_index")] public int LayerIndex { get; set; }
        [JsonPropertyName("layer_name")] public string LayerName { get; set; } = string.Empty;
        [JsonPropertyName("layer_type")] public string LayerType { get; set; } = string.Empty;
        [JsonPropertyName("mean")] public double Mean { get; set; }
        [JsonPropertyName("std")] public double Std { get; set; }
        [JsonPropertyName("min")] public double Min { get; set; }
        [JsonPropertyName("max")] public double Max { get; set; }
        [JsonPropertyName("sparsity")] public double Sparsity { get; set; }
        [JsonPropertyName("dead_neurons")] public int DeadNeurons { get; set; }
        [JsonPropertyName("grad_norm")] public double GradNorm { get; set; }
    }
}
