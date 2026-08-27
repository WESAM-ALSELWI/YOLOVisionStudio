using System;

namespace YOLOVisionStudio.Models.Domain
{
    public enum ModelStatus
    {
        Training,
        Ready,
        Failed
    }

    /// <summary>
    /// Represents a trained YOLO model.
    /// </summary>
    public class YoloModel
    {
        public int Id { get; set; }
        public int ProjectId { get; set; }
        
        public string Name { get; set; } = string.Empty;
        public string YoloVersion { get; set; } = string.Empty; // v5, v6, v7, v8, v9, v10, v11
        public ObjectiveType Task { get; set; }
        
        public string WeightsPath { get; set; } = string.Empty;
        public string ModelSize { get; set; } = "medium"; // nano, small, medium, large, xlarge
        
        public long Parameters { get; set; }
        public double FLOPs { get; set; }
        public int InputSize { get; set; }
        
        public double mAP50 { get; set; }
        public double mAP5095 { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double FPS { get; set; }
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public ModelStatus Status { get; set; } = ModelStatus.Ready;
        public string? Description { get; set; }
        
        public string? ExportedFormats { get; set; } // JSON list of available exports

        // Navigation
        public Project? Project { get; set; }
    }
}
