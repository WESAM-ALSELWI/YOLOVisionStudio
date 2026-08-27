using System;

namespace YOLOVisionStudio.Models.Domain
{
    public enum InputType
    {
        Image,
        Video,
        Camera
    }

    /// <summary>
    /// Represents an inference prediction record.
    /// </summary>
    public class Prediction
    {
        public int Id { get; set; }
        public int ModelId { get; set; }
        public int ProjectId { get; set; }
        
        public string InputPath { get; set; } = string.Empty;
        public string OutputPath { get; set; } = string.Empty;
        
        public InputType InputType { get; set; }
        public string? DetectedObjects { get; set; } // JSON
        
        public long ProcessingTimeMs { get; set; }
        public double FPS { get; set; }
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public string CreatedByUserId { get; set; } = string.Empty;

        // Navigation
        public YoloModel? Model { get; set; }
        public Project? Project { get; set; }
    }
}
