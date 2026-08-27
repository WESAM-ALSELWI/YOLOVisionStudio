using System;
using System.Collections.Generic;

namespace YOLOVisionStudio.Models.Domain
{
    /// <summary>Supported objective types for YOLO.</summary>
    public enum ObjectiveType
    {
        Detection,
        Segmentation,
        Pose,
        Classification
    }

    /// <summary>Project status.</summary>
    public enum ProjectStatus
    {
        Active,
        Archived,
        Draft
    }

    /// <summary>
    /// Represents a YOLO training or inference project.
    /// </summary>
    public class Project
    {
        /// <summary>Gets or sets the project ID.</summary>
        public int Id { get; set; }

        /// <summary>Gets or sets the name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Gets or sets the description.</summary>
        public string? Description { get; set; }

        /// <summary>Gets or sets the objective type.</summary>
        public ObjectiveType ObjectiveType { get; set; }

        /// <summary>Gets or sets the project status.</summary>
        public ProjectStatus Status { get; set; } = ProjectStatus.Active;

        /// <summary>Gets or sets the creation date.</summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>Gets or sets the last updated date.</summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>Gets or sets the ID of the user who created the project.</summary>
        public string CreatedByUserId { get; set; } = "user-admin";

        /// <summary>Gets or sets the URL of the project's thumbnail.</summary>
        public string? ThumbnailUrl { get; set; }

        /// <summary>Gets or sets tags as a JSON string.</summary>
        public string? Tags { get; set; } // JSON string

        /// <summary>Gets or sets the dataset count.</summary>
        public int DatasetCount { get; set; }

        /// <summary>Gets or sets the model count.</summary>
        public int ModelCount { get; set; }

        /// <summary>Gets or sets the experiment count.</summary>
        public int ExperimentCount { get; set; }

        // Navigation properties
        /// <summary>The user who created the project.</summary>
        public AppUser? CreatedByUser { get; set; }

        /// <summary>The datasets associated with this project.</summary>
        public List<Dataset> Datasets { get; set; } = new();

        /// <summary>The models associated with this project.</summary>
        public List<YoloModel> Models { get; set; } = new();

        /// <summary>The experiments associated with this project.</summary>
        public List<Experiment> Experiments { get; set; } = new();
    }
}
