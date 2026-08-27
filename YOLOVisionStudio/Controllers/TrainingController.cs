using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;
using YOLOVisionStudio.Models.ViewModels;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class TrainingController : Controller
    {
        private readonly ITrainingService _trainingService;
        private readonly IDatasetService _datasetService;

        public TrainingController(ITrainingService trainingService, IDatasetService datasetService)
        {
            _trainingService = trainingService;
            _datasetService = datasetService;
        }

        /// <summary>Staged training wizard: 1) Dataset → 2) Model → 3) Settings → Train.</summary>
        [HttpGet]
        public async Task<IActionResult> Wizard(int projectId = 1, int datasetId = 0)
        {
            var all = await _datasetService.GetDatasetsForProjectAsync(projectId);

            // Classify each dataset: usable, needs splitting into the YOLO
            // structure, or completely broken (no images at all → removed).
            var items = new List<WizardDatasetItem>();
            foreach (var ds in all)
            {
                var path = ds.StoragePath ?? string.Empty;
                var detected = CountImagesRecursively(path);
                if (string.IsNullOrWhiteSpace(path) || !System.IO.Directory.Exists(path) || detected == 0)
                {
                    await _datasetService.DeleteDatasetAsync(ds.Id);
                    continue;
                }
                items.Add(new WizardDatasetItem
                {
                    Dataset = ds,
                    NeedsSplit = !IsValidYoloDataset(path),
                    DetectedImages = detected
                });
            }

            return View(new WizardViewModel
            {
                ProjectId = projectId,
                DatasetId = datasetId,
                Datasets = items
            });
        }

        private static int CountImagesRecursively(string? path)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(path) || !System.IO.Directory.Exists(path)) return 0;
                return System.IO.Directory.EnumerateFiles(path, "*", System.IO.SearchOption.AllDirectories)
                    .Count(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase));
            }
            catch { return 0; }
        }

        /// <summary>A usable YOLO dataset needs a data.yaml AND at least one real training image.</summary>
        private static bool IsValidYoloDataset(string? storagePath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(storagePath) || !System.IO.Directory.Exists(storagePath))
                    return false;
                if (!System.IO.File.Exists(System.IO.Path.Combine(storagePath, "data.yaml")))
                    return false;

                var trainDir = System.IO.Path.Combine(storagePath, "images", "train");
                if (!System.IO.Directory.Exists(trainDir))
                    return false;

                return System.IO.Directory.EnumerateFiles(trainDir)
                    .Any(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                           || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                           || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase));
            }
            catch { return false; }
        }

        [HttpGet]
        public IActionResult Configure(int projectId, int datasetId)
        {
            return View(new TrainingConfigViewModel { ProjectId = projectId, DatasetId = datasetId });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Start(TrainingConfigViewModel model)
        {
            // Server-side validation before anything is downloaded or started.
            if (model.DatasetId <= 0)
                ModelState.AddModelError(nameof(model.DatasetId), "Select a dataset first (use a sample dataset or upload your own).");
            if (string.IsNullOrWhiteSpace(model.BaseModel))
                ModelState.AddModelError(nameof(model.BaseModel), "Select a model architecture and scale.");

            if (!ModelState.IsValid)
                return View("Configure", model);

            var job = new TrainingJob
            {
                ProjectId = model.ProjectId,
                DatasetId = model.DatasetId,
                Name = $"Training_{System.DateTime.UtcNow.Ticks}",
                YoloVersion = model.YoloVersion,
                Task = System.Enum.TryParse<ObjectiveType>(model.Task, true, out var objType) ? objType : ObjectiveType.Detection,
                BaseModel = model.BaseModel,
                Epochs = model.Epochs,
                BatchSize = model.BatchSize,
                ImageSize = model.ImageSize,
                Optimizer = string.IsNullOrWhiteSpace(model.Optimizer) ? "auto" : model.Optimizer,
                LearningRate = model.LearningRate,
                WeightDecay = model.WeightDecay,
                AugmentationConfig = System.Text.Json.JsonSerializer.Serialize(new
                {
                    mosaic = model.Mosaic,
                    mixup = model.MixUp
                })
            };
            await _trainingService.StartJobAsync(job);

            if (job.Status == JobStatus.Failed)
            {
                TempData["ErrorMessage"] = job.ErrorMessage ?? "Failed to start the training job on the AI Engine.";
                return RedirectToAction("Configure", new { projectId = model.ProjectId, datasetId = model.DatasetId });
            }

            return RedirectToAction("Monitor", new { id = job.Id });
        }

        [HttpGet]
        public async Task<IActionResult> Monitor(int id)
        {
            var job = await _trainingService.GetJobByIdAsync(id);
            if (job == null) return NotFound();
            return View(new TrainingMonitorViewModel { Job = job });
        }

        public async Task<IActionResult> History()
        {
            var jobs = await _trainingService.GetAllJobsAsync();
            return View(jobs);
        }
    }
}
