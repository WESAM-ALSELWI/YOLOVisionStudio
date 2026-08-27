using System;

namespace YOLOVisionStudio.Models.Domain
{
    public enum JobStatus
    {
        Pending,
        Running,
        Completed,
        Failed,
        Cancelled
    }

    /// <summary>
    /// Represents a background training job.
    /// </summary>
    public class TrainingJob
    {
        public int Id { get; set; }
        public int ProjectId { get; set; }
        public int DatasetId { get; set; }
        public int? ModelId { get; set; }
        public int? ExperimentId { get; set; }
        
        public string Name { get; set; } = string.Empty;
        public JobStatus Status { get; set; } = JobStatus.Pending;
        
        public string YoloVersion { get; set; } = string.Empty;
        public ObjectiveType Task { get; set; }
        public string BaseModel { get; set; } = string.Empty;
        
        public int Epochs { get; set; }
        public int ImageSize { get; set; }
        public int BatchSize { get; set; }
        
        public string Optimizer { get; set; } = "auto";
        public double LearningRate { get; set; }
        public double WeightDecay { get; set; }
        public double Momentum { get; set; }
        public string Scheduler { get; set; } = "auto";
        public int WarmupEpochs { get; set; }
        public int FreezeLayersCount { get; set; }
        
        public string? AugmentationConfig { get; set; } // JSON
        
        public int CurrentEpoch { get; set; }
        public double BestmAP50 { get; set; }
        public double BestmAP5095 { get; set; }
        
        public DateTime? StartedAt { get; set; }
        public DateTime? CompletedAt { get; set; }
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public string CreatedByUserId { get; set; } = "user-admin";
        
        public string? LogPath { get; set; }
        public string? ResultsPath { get; set; }
        public string? ErrorMessage { get; set; }
        
        public string? PythonJobId { get; set; }
        public string? OutputModelPath { get; set; }

        // Navigation
        public Project? Project { get; set; }
        public Dataset? Dataset { get; set; }
        public AppUser? CreatedByUser { get; set; }
    }
}
