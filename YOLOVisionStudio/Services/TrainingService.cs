using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using YOLOVisionStudio.Data;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public class TrainingService : ITrainingService
    {
        private readonly ApplicationDbContext _context;
        private readonly IYoloAiService _aiService;

        public TrainingService(ApplicationDbContext context, IYoloAiService aiService)
        {
            _context = context;
            _aiService = aiService;
        }

        public async Task<TrainingJob> StartJobAsync(TrainingJob job)
        {
            _context.TrainingJobs.Add(job);
            await _context.SaveChangesAsync();

            // Build the real payload expected by the FastAPI engine.
            var dataset = await _context.Datasets.FindAsync(job.DatasetId);
            var yamlPath = Path.Combine(dataset?.StoragePath ?? string.Empty, "data.yaml");

            var request = new TrainingRequest
            {
                JobId = job.Id,
                ProjectId = job.ProjectId.ToString(),
                DatasetYamlPath = yamlPath,
                Model = string.IsNullOrWhiteSpace(job.BaseModel) ? "yolo11n.pt" : job.BaseModel,
                Task = job.Task.ToString().ToLowerInvariant(),
                Epochs = job.Epochs,
                Imgsz = job.ImageSize,
                Batch = job.BatchSize,
                Optimizer = job.Optimizer ?? "auto",
                Lr0 = job.LearningRate <= 0 ? 0.01 : job.LearningRate,
                WeightDecay = job.WeightDecay,
                Name = $"job{job.Id}_{DateTime.UtcNow:yyyyMMdd_HHmmss}",
                Device = null, // let ultralytics pick CPU/GPU automatically
            };

            if (!string.IsNullOrEmpty(job.AugmentationConfig))
            {
                try
                {
                    using var doc = System.Text.Json.JsonDocument.Parse(job.AugmentationConfig);
                    var dict = new Dictionary<string, object>();
                    foreach (var p in doc.RootElement.EnumerateObject())
                        dict[p.Name] = p.Value.TryGetDouble(out var d) ? d : p.Value.ToString();
                    request.AugmentParams = dict;
                }
                catch { /* augmentation config is optional */ }
            }

            var res = await _aiService.StartTraining(request);
            if (res.Success)
            {
                job.PythonJobId = res.PythonJobId;
                job.Status = JobStatus.Running;
                job.StartedAt = DateTime.UtcNow;
            }
            else
            {
                job.Status = JobStatus.Failed;
                job.ErrorMessage = string.IsNullOrWhiteSpace(res.Message)
                    ? "AI Engine did not accept the training request."
                    : res.Message;
            }
            await _context.SaveChangesAsync();

            return job;
        }

        public async Task<TrainingJob?> GetJobByIdAsync(int id)
        {
            return await _context.TrainingJobs.FindAsync(id);
        }

        public async Task<List<TrainingJob>> GetAllJobsAsync()
        {
            return await _context.TrainingJobs.ToListAsync();
        }
    }
}
