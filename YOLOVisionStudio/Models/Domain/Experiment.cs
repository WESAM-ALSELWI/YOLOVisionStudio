using System;
using System.Collections.Generic;

namespace YOLOVisionStudio.Models.Domain
{
    public enum ExperimentStatus
    {
        Active,
        Completed,
        Failed
    }

    /// <summary>
    /// Represents an experiment that tracks training runs and comparisons.
    /// </summary>
    public class Experiment
    {
        public int Id { get; set; }
        public int ProjectId { get; set; }
        
        public string Name { get; set; } = string.Empty;
        public string? Description { get; set; }
        public ExperimentStatus Status { get; set; } = ExperimentStatus.Active;
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        public string? Tags { get; set; } // JSON
        public string? Notes { get; set; }
        
        public string? BestMetricName { get; set; }
        public double BestMetricValue { get; set; }

        // Navigation
        public Project? Project { get; set; }
        public List<TrainingJob> TrainingJobs { get; set; } = new();
        public List<TrainingMetric> Metrics { get; set; } = new();
    }
}
