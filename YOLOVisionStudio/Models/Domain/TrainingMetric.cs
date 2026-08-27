using System;

namespace YOLOVisionStudio.Models.Domain
{
    /// <summary>
    /// Represents a metric collected during a training epoch.
    /// </summary>
    public class TrainingMetric
    {
        public int Id { get; set; }
        public int TrainingJobId { get; set; }
        public int? ExperimentId { get; set; }
        
        public int Epoch { get; set; }
        
        public double BoxLoss { get; set; }
        public double ClassLoss { get; set; }
        public double DFLLoss { get; set; }
        
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double mAP50 { get; set; }
        public double mAP5095 { get; set; }
        
        public double LearningRate { get; set; }
        
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        // Navigation
        public TrainingJob? TrainingJob { get; set; }
        public Experiment? Experiment { get; set; }
    }
}
