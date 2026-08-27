using System;

namespace YOLOVisionStudio.Models.Domain
{
    public enum LogLevel
    {
        Info,
        Warning,
        Error
    }

    /// <summary>
    /// Represents a system log entry.
    /// </summary>
    public class SystemLog
    {
        public int Id { get; set; }
        public LogLevel Level { get; set; }
        public string Message { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string? UserId { get; set; }
        public string? Details { get; set; }
    }
}
