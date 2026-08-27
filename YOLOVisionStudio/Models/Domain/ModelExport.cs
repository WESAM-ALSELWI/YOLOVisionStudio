using System;

namespace YOLOVisionStudio.Models.Domain
{
    public enum ExportFormat
    {
        ONNX,
        TensorRT,
        OpenVINO,
        TorchScript,
        CoreML
    }

    public enum ExportStatus
    {
        Pending,
        Processing,
        Completed,
        Failed
    }

    /// <summary>
    /// Represents an exported model configuration.
    /// </summary>
    public class ModelExport
    {
        public int Id { get; set; }
        public int ModelId { get; set; }
        
        public ExportFormat Format { get; set; }
        public DateTime ExportedAt { get; set; } = DateTime.UtcNow;
        
        public string? FilePath { get; set; }
        public long FileSize { get; set; }
        
        public ExportStatus Status { get; set; } = ExportStatus.Pending;
        public string? ErrorMessage { get; set; }

        // Navigation
        public YoloModel? Model { get; set; }
    }
}
