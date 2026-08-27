using System;
using System.Collections.Generic;

namespace YOLOVisionStudio.Models.Domain
{
    public enum DatasetFormat
    {
        YOLO_TXT,
        COCO_JSON,
        VOC_XML,
        LabelMe
    }

    public enum DatasetStatus
    {
        Processing,
        Ready,
        Error
    }

    /// <summary>
    /// Represents a dataset used for training or validation.
    /// </summary>
    public class Dataset
    {
        public int Id { get; set; }
        public int ProjectId { get; set; }
        public string Name { get; set; } = string.Empty;
        public string? Description { get; set; }
        public DatasetFormat Format { get; set; }
        
        public int TotalImages { get; set; }
        public int TotalLabeled { get; set; }
        public int TotalUnlabeled { get; set; }
        
        public string? Classes { get; set; } // JSON string list
        public string? ClassDistribution { get; set; } // JSON
        
        public string StoragePath { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        public DatasetStatus Status { get; set; } = DatasetStatus.Processing;
        
        public long SizeBytes { get; set; }
        public int TrainCount { get; set; }
        public int ValCount { get; set; }
        public int TestCount { get; set; }

        // Navigation
        public Project? Project { get; set; }
        public List<DatasetImage> Images { get; set; } = new();
    }
}
