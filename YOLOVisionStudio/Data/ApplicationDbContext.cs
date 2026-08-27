using Microsoft.EntityFrameworkCore;
using System;
using System.Threading;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Data
{
    /// <summary>
    /// Database context for the application.
    /// </summary>
    public class ApplicationDbContext : DbContext
    {
        public DbSet<AppUser> Users { get; set; } = null!;
        public DbSet<Project> Projects { get; set; } = null!;
        public DbSet<Dataset> Datasets { get; set; } = null!;
        public DbSet<DatasetImage> DatasetImages { get; set; } = null!;
        public DbSet<YoloModel> Models { get; set; } = null!;
        public DbSet<TrainingJob> TrainingJobs { get; set; } = null!;
        public DbSet<Experiment> Experiments { get; set; } = null!;
        public DbSet<TrainingMetric> TrainingMetrics { get; set; } = null!;
        public DbSet<Prediction> Predictions { get; set; } = null!;
        public DbSet<ModelExport> ModelExports { get; set; } = null!;
        public DbSet<SystemLog> SystemLogs { get; set; } = null!;

        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        protected override void OnModelCreating(ModelBuilder builder)
        {
            // Seed Default Local User
            builder.Entity<AppUser>().HasData(
                new AppUser
                {
                    Id = "user-admin",
                    UserName = "admin",
                    Email = "admin@yolovision.studio",
                    Role = UserRole.Admin,
                    DisplayName = "System Administrator",
                    CreatedAt = DateTime.UtcNow
                }
            );

            // Seed Default Demo Project for Local Mode
            builder.Entity<Project>().HasData(
                new Project
                {
                    Id = 1,
                    Name = "Demo YOLO Vision Project",
                    Description = "Default local workspace project for YOLO operations",
                    ObjectiveType = ObjectiveType.Detection,
                    Status = ProjectStatus.Active,
                    CreatedByUserId = "user-admin",
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                }
            );

            // Foreign Keys and Relationships (Optional for Local Mode)
            builder.Entity<Project>()
                .HasOne(p => p.CreatedByUser)
                .WithMany()
                .HasForeignKey(p => p.CreatedByUserId)
                .IsRequired(false)
                .OnDelete(DeleteBehavior.SetNull);

            builder.Entity<TrainingJob>()
                .HasOne(t => t.CreatedByUser)
                .WithMany()
                .HasForeignKey(t => t.CreatedByUserId)
                .IsRequired(false)
                .OnDelete(DeleteBehavior.SetNull);
        }

        public override async Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
        {
            var entries = ChangeTracker.Entries();
            foreach (var entry in entries)
            {
                if (entry.State == EntityState.Added || entry.State == EntityState.Modified)
                {
                    if (entry.Entity is Project project)
                        project.UpdatedAt = DateTime.UtcNow;
                    else if (entry.Entity is Dataset dataset)
                        dataset.UpdatedAt = DateTime.UtcNow;
                    else if (entry.Entity is Experiment experiment)
                        experiment.UpdatedAt = DateTime.UtcNow;
                }
            }
                return await base.SaveChangesAsync(cancellationToken);
        }
    }
}