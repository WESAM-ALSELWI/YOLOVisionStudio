using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Http;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Models.ViewModels
{
    public class DashboardViewModel
    {
        public int ProjectCount { get; set; }
        public int DatasetCount { get; set; }
        public int ModelCount { get; set; }
        public int ActiveJobs { get; set; }
        public List<Project> RecentProjects { get; set; } = new();
        public List<TrainingJob> RecentJobs { get; set; } = new();
        public MonitorStats? SystemStats { get; set; }
    }

    public class ProjectViewModel
    {
        public Project? Project { get; set; }
    }

    public class ProjectCreateViewModel
    {
        public string Name { get; set; } = string.Empty;
        public string? Description { get; set; }
        public ObjectiveType ObjectiveType { get; set; }
    }

    public class DatasetViewModel
    {
        public Dataset? Dataset { get; set; }
        public IFormFile? UploadFile { get; set; }
    }

    public class DatasetAnalysisViewModel { public Dataset? Dataset { get; set; } }
    
    public class WizardDatasetItem
    {
        public Dataset Dataset { get; set; } = new();
        public bool NeedsSplit { get; set; }
        public int DetectedImages { get; set; }
    }

    public class WizardViewModel
    {
        public int ProjectId { get; set; } = 1;
        public int DatasetId { get; set; }
        public List<WizardDatasetItem> Datasets { get; set; } = new();
    }

    public class SplitDatasetViewModel
    {
        public Dataset Dataset { get; set; } = new();
        public int DetectedImages { get; set; }
        public bool IsYoloStructured { get; set; }
        public string ReturnTo { get; set; } = "index";
    }

    /// <summary>A dataset folder found on disk under wwwroot/datasets.</summary>
    public class DiskDatasetItem
    {
        public string Name { get; set; } = string.Empty;
        public string Path { get; set; } = string.Empty;
        public bool IsYoloStructured { get; set; }
        public int TotalImages { get; set; }
        public int TrainImages { get; set; }
        public int? DatasetId { get; set; }
        public string Status => IsYoloStructured ? "Ready" : "Needs Split";
    }

    public class DatasetIndexViewModel
    {
        public int ProjectId { get; set; } = 1;
        public List<Dataset> Registered { get; set; } = new();
        public List<DiskDatasetItem> OnDisk { get; set; } = new();
    }

    public class TrainingConfigViewModel
    {
        public int ProjectId { get; set; }
        public int DatasetId { get; set; }
        public string YoloVersion { get; set; } = "v11";
        public string Task { get; set; } = "detect";
        public string BaseModel { get; set; } = "yolo11n.pt";
        [Range(1, 2000, ErrorMessage = "Epochs must be between 1 and 2000.")]
        public int Epochs { get; set; } = 100;
        [Range(32, 1536, ErrorMessage = "Image size must be between 32 and 1536 px.")]
        public int ImageSize { get; set; } = 640;
        [Range(1, 128, ErrorMessage = "Batch size must be between 1 and 128.")]
        public int BatchSize { get; set; } = 8;

        // Advanced hyperparameters
        public string Optimizer { get; set; } = "auto";
        [Range(0.00001, 1.0, ErrorMessage = "Learning rate must be between 0.00001 and 1.")]
        public double LearningRate { get; set; } = 0.01;
        [Range(0, 0.1, ErrorMessage = "Weight decay must be between 0 and 0.1.")]
        public double WeightDecay { get; set; } = 0.0005;
        [Range(0, 1, ErrorMessage = "Mosaic must be between 0 and 1.")]
        public double Mosaic { get; set; } = 1.0;
        [Range(0, 1, ErrorMessage = "MixUp must be between 0 and 1.")]
        public double MixUp { get; set; } = 0.0;
    }
    
    public class TrainingMonitorViewModel { public TrainingJob? Job { get; set; } }
    public class ModelComparisonViewModel { public List<YoloModel> Models { get; set; } = new(); }
    public class ExperimentViewModel { public Experiment? Experiment { get; set; } }
    public class ExportViewModel { public int ModelId { get; set; } public string Format { get; set; } = string.Empty; }
    public class AssistantViewModel { public List<ChatMessage> Messages { get; set; } = new(); }
    
    public class ChatMessage
    {
        public string Role { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
    }

    public class MonitorViewModel
    {
        public double CpuUsage { get; set; }
        public double RamUsageMb { get; set; }
        public double GpuUsage { get; set; }
    }

    public class MonitorStats
    {
        public double CpuUsagePercentage { get; set; }
        public double RamUsageMB { get; set; }
        public double TotalRamMB { get; set; }
        public double GpuUsagePercentage { get; set; }
        public string? GpuName { get; set; }
    }
}
