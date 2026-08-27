using System;

namespace YOLOVisionStudio.Models.Domain
{
    /// <summary>
    /// Represents an image within a dataset.
    /// </summary>
    public class DatasetImage
    {
        public int Id { get; set; }
        public int DatasetId { get; set; }
        
        public string FileName { get; set; } = string.Empty;
        public string FilePath { get; set; } = string.Empty;
        
        public int Width { get; set; }
        public int Height { get; set; }
        public long FileSize { get; set; }
        
        public bool HasAnnotation { get; set; }
        public string? AnnotationPath { get; set; }
        
        public bool IsDuplicate { get; set; }
        public bool IsCorrupted { get; set; }
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Navigation
        public Dataset? Dataset { get; set; }
    }
}
