using Microsoft.AspNetCore.Mvc;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;
using YOLOVisionStudio.Models.ViewModels;
using YOLOVisionStudio.Services;
using Microsoft.AspNetCore.Http;
using System.Collections.Generic;

namespace YOLOVisionStudio.Controllers
{
    public class DatasetController : Controller
    {
        private readonly IDatasetService _datasetService;
        private readonly IYoloAiService _aiService;

        public DatasetController(IDatasetService datasetService, IYoloAiService aiService)
        {
            _datasetService = datasetService;
            _aiService = aiService;
        }

        public async Task<IActionResult> Index(int projectId = 1)
        {
            var datasets = await _datasetService.GetDatasetsForProjectAsync(projectId);

            // Every folder under wwwroot/datasets is treated as a dataset
            // (zip archives are ignored). Structure is checked locally so this
            // works even when the AI engine is not running yet.
            var onDisk = new List<DiskDatasetItem>();
            var root = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "datasets");
            if (Directory.Exists(root))
            {
                foreach (var dir in Directory.GetDirectories(root))
                {
                    var total = CountImages(dir);
                    if (total == 0) continue;
                    var trainDir = Path.Combine(dir, "images", "train");
                    var trainImages = Directory.Exists(trainDir) ? CountImages(trainDir) : 0;
                    onDisk.Add(new DiskDatasetItem
                    {
                        Name = Path.GetFileName(dir),
                        Path = dir,
                        IsYoloStructured = System.IO.File.Exists(Path.Combine(dir, "data.yaml")) && trainImages > 0,
                        TotalImages = total,
                        TrainImages = trainImages,
                        DatasetId = datasets.FirstOrDefault(d =>
                            string.Equals(d.StoragePath, dir, StringComparison.OrdinalIgnoreCase))?.Id
                    });
                }
            }

            return View(new DatasetIndexViewModel
            {
                ProjectId = projectId,
                Registered = datasets,
                OnDisk = onDisk
            });
        }

        private static int CountImages(string path)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path)) return 0;
                return Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories)
                    .Count(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)
                             || f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase));
            }
            catch { return 0; }
        }

        [HttpGet]
        public IActionResult Upload(int projectId = 1)
        {
            ViewBag.ProjectId = projectId;
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Upload(int projectId, string name, IFormFile file,
            int trainPct = 70, int valPct = 15, int testPct = 15)
        {
            ViewBag.ProjectId = projectId;

            if (file == null || file.Length == 0)
            {
                ModelState.AddModelError(string.Empty, "Please select a dataset archive (.zip) to upload.");
                return View();
            }

            // Sanitize the dataset folder name.
            var safeName = string.Join("_", (name ?? "dataset").Split(Path.GetInvalidFileNameChars()));
            if (string.IsNullOrWhiteSpace(safeName)) safeName = "dataset";

            var destDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads", "datasets", safeName);

            try
            {
                // Extract the archive into its own folder (originals are kept).
                Directory.CreateDirectory(destDir);
                await using (var stream = file.OpenReadStream())
                {
                    System.IO.Compression.ZipFile.ExtractToDirectory(stream, destDir, overwriteFiles: true);
                }

                // If the zip wrapped everything in one root folder, use that as the dataset root.
                var entries = Directory.GetFileSystemEntries(destDir);
                if (entries.Length == 1 && Directory.Exists(entries[0]))
                    destDir = entries[0];

                // Check whether the extracted content already follows the YOLO structure.
                var status = await _aiService.GetSplitStatus(destDir);

                var dataset = new Dataset
                {
                    ProjectId = projectId,
                    Name = name ?? safeName,
                    Format = DatasetFormat.YOLO_TXT,
                    StoragePath = destDir,
                    Status = DatasetStatus.Processing
                };

                if (status.IsYoloStructured)
                {
                    dataset.TotalImages = status.TrainImages;
                    dataset.TotalLabeled = status.TrainImages;
                    dataset.Status = DatasetStatus.Ready;
                    await _datasetService.CreateDatasetAsync(dataset);
                    TempData["SuccessMessage"] = $"Dataset '{dataset.Name}' uploaded ({status.TrainImages} images). It already follows the YOLO structure.";
                    return RedirectToAction("Index", new { projectId });
                }

                // Not YOLO-structured -> validate ratios and split automatically.
                if (trainPct + valPct + testPct != 100 || trainPct <= 0)
                {
                    ModelState.AddModelError(string.Empty,
                        $"The split must total exactly 100% with a positive train share (you entered {trainPct} + {valPct} + {testPct}).");
                    return View();
                }

                var result = await _aiService.SplitDataset(new SplitDatasetRequest
                {
                    DatasetPath = destDir,
                    TrainPct = trainPct,
                    ValPct = valPct,
                    TestPct = testPct
                });

                if (!result.Success)
                {
                    ModelState.AddModelError(string.Empty, $"Upload succeeded but automatic split failed: {result.Message}");
                    return View();
                }

                dataset.TotalImages = result.TotalImages;
                dataset.TotalLabeled = result.LabelsFound;
                dataset.TotalUnlabeled = result.TotalImages - result.LabelsFound;
                dataset.Classes = System.Text.Json.JsonSerializer.Serialize(
                    Enumerable.Range(0, Math.Max(1, result.Classes)).Select(i => $"class_{i}").ToList());
                dataset.TrainCount = result.Counts.TryGetValue("train", out var tr) ? tr : 0;
                dataset.ValCount = result.Counts.TryGetValue("val", out var va) ? va : 0;
                dataset.TestCount = result.Counts.TryGetValue("test", out var te) ? te : 0;
                dataset.Status = DatasetStatus.Ready;

                await _datasetService.CreateDatasetAsync(dataset);

                TempData["SuccessMessage"] = $"{result.Message} Split ratio: {trainPct}/{valPct}/{testPct}.";
                return RedirectToAction("Index", new { projectId });
            }
            catch (System.IO.InvalidDataException)
            {
                ModelState.AddModelError(string.Empty, "The uploaded file is not a valid .zip archive.");
                return View();
            }
            catch (Exception ex)
            {
                ModelState.AddModelError(string.Empty, $"Upload failed: {ex.Message}");
                return View();
            }
        }

        [HttpGet]
        public async Task<IActionResult> Analyze(int id = 1)
        {
            var dataset = await _datasetService.GetDatasetByIdAsync(id);
            if (dataset == null)
            {
                // Create fallback dataset object for visual demonstration
                dataset = new Dataset
                {
                    Id = id,
                    Name = "Helmet Safety Sample Dataset",
                    TotalImages = 500,
                    TotalLabeled = 480,
                    TotalUnlabeled = 20,
                    Classes = "[\"Helmet\", \"No-Helmet\"]",
                    TrainCount = 400,
                    ValCount = 80,
                    TestCount = 20,
                    Format = DatasetFormat.YOLO_TXT,
                    Status = DatasetStatus.Ready
                };
            }
            return View(dataset);
        }

        [HttpGet]
        public IActionResult Augment()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> InstallSample(string key, int projectId = 1)
        {
            // Ask the AI engine to install the sample dataset. It reuses the
            // local copy in wwwroot/datasets when present, otherwise downloads.
            var result = await _aiService.InstallSampleDataset(key ?? string.Empty);
            if (!result.Success || !result.Installed)
            {
                TempData["SuccessMessage"] = null;
                ModelState.AddModelError(string.Empty, result.Message);
                return RedirectToAction("Index", new { projectId });
            }

            // Reuse the existing DB record when this dataset was installed before.
            var datasets = await _datasetService.GetDatasetsForProjectAsync(projectId);
            var dataset = datasets.FirstOrDefault(d =>
                string.Equals(d.StoragePath, result.StoragePath, StringComparison.OrdinalIgnoreCase));

            if (dataset == null)
            {
                var classesJson = System.Text.Json.JsonSerializer.Serialize(
                    Enumerable.Range(0, Math.Max(1, result.ClassesCount)).Select(i => $"class_{i}").ToList());
                dataset = new Dataset
                {
                    ProjectId = projectId,
                    Name = result.DatasetName,
                    Description = $"{result.Difficulty} sample dataset - {result.Message}",
                    Format = DatasetFormat.YOLO_TXT,
                    StoragePath = result.StoragePath,
                    TotalImages = result.Images,
                    TotalLabeled = result.Images,
                    Classes = classesJson,
                    TrainCount = (int)(result.Images * 0.75),
                    ValCount = Math.Max(1, result.Images - (int)(result.Images * 0.75)),
                    Status = DatasetStatus.Ready
                };
                await _datasetService.CreateDatasetAsync(dataset);
            }

            TempData["SuccessMessage"] = $"{result.DatasetName}: {result.Message} Ready to train!";
            return RedirectToAction("Wizard", "Training", new { projectId, datasetId = dataset.Id });
        }

        /// <summary>Split / re-split tool: builds the YOLO images/{train,val,test}
        /// + labels structure with user-chosen ratios (default 70/15/15).
        /// Works with a DB dataset (id) or any folder under wwwroot/datasets (name).</summary>
        [HttpGet]
        public async Task<IActionResult> Split(int id = 0, string? name = null, string returnTo = "index")
        {
            var (dataset, path, resolvedName) = await ResolveDatasetAsync(id, name);
            if (path == null || !Directory.Exists(path))
                return NotFound();

            var status = await _aiService.GetSplitStatus(path);

            return View(new SplitDatasetViewModel
            {
                Dataset = new Dataset
                {
                    Id = dataset?.Id ?? 0,
                    ProjectId = dataset?.ProjectId ?? 1,
                    Name = dataset?.Name ?? resolvedName ?? Path.GetFileName(path),
                    StoragePath = path
                },
                DetectedImages = status.UnsplitImages > 0 ? status.UnsplitImages : status.TrainImages,
                IsYoloStructured = status.IsYoloStructured,
                ReturnTo = returnTo
            });
        }

        private async Task<(Dataset? dataset, string? path, string? name)> ResolveDatasetAsync(int id, string? name)
        {
            if (id > 0)
            {
                var ds = await _datasetService.GetDatasetByIdAsync(id);
                if (ds != null && !string.IsNullOrWhiteSpace(ds.StoragePath))
                    return (ds, ds.StoragePath, ds.Name);
            }

            var safeName = string.Join("_", (name ?? string.Empty).Split(Path.GetInvalidFileNameChars()));
            if (!string.IsNullOrWhiteSpace(safeName))
            {
                var dir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "datasets", safeName);
                if (Directory.Exists(dir))
                {
                    var existing = (await _datasetService.GetDatasetsForProjectAsync(1)).FirstOrDefault(d =>
                        string.Equals(d.StoragePath, dir, StringComparison.OrdinalIgnoreCase));
                    return (existing, dir, safeName);
                }
            }
            return (null, null, null);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> SplitDataset(int id = 0, string? name = null, int projectId = 1,
            int trainPct = 70, int valPct = 15, int testPct = 15, string returnTo = "index")
        {
            var (existing, path, _) = await ResolveDatasetAsync(id, name);
            if (path == null)
                return NotFound();

            if (trainPct + valPct + testPct != 100 || trainPct <= 0)
            {
                TempData["SplitError"] = $"The split must total exactly 100% with a positive train share (you entered {trainPct} + {valPct} + {testPct}).";
                return RedirectToAction("Split", new { id, name, returnTo });
            }

            var result = await _aiService.SplitDataset(new SplitDatasetRequest
            {
                DatasetPath = path,
                TrainPct = trainPct,
                ValPct = valPct,
                TestPct = testPct
            });

            if (!result.Success)
            {
                TempData["SplitError"] = result.Message;
                return RedirectToAction("Split", new { id, name, returnTo });
            }

            // Update the existing record or register this folder as a new dataset.
            var dataset = existing ?? new Dataset
            {
                ProjectId = projectId,
                Name = Path.GetFileName(path),
                Format = DatasetFormat.YOLO_TXT,
                StoragePath = path
            };

            dataset.TotalImages = result.TotalImages;
            dataset.TotalLabeled = result.LabelsFound;
            dataset.TotalUnlabeled = result.TotalImages - result.LabelsFound;
            dataset.Classes = System.Text.Json.JsonSerializer.Serialize(
                Enumerable.Range(0, Math.Max(1, result.Classes)).Select(i => $"class_{i}").ToList());
            dataset.TrainCount = result.Counts.TryGetValue("train", out var tr) ? tr : 0;
            dataset.ValCount = result.Counts.TryGetValue("val", out var va) ? va : 0;
            dataset.TestCount = result.Counts.TryGetValue("test", out var te) ? te : 0;
            dataset.Status = DatasetStatus.Ready;
            dataset.UpdatedAt = DateTime.UtcNow;
            if (existing != null)
                await _datasetService.UpdateDatasetAsync(dataset);
            else
                await _datasetService.CreateDatasetAsync(dataset);

            TempData["SuccessMessage"] = $"{result.Message} Split ratio: {trainPct}/{valPct}/{testPct}.";
            if (returnTo == "wizard")
                return RedirectToAction("Wizard", "Training", new { projectId = dataset.ProjectId, datasetId = dataset.Id });
            return RedirectToAction("Index", new { projectId = dataset.ProjectId });
        }

        /// <summary>Register an already YOLO-structured folder from
        /// wwwroot/datasets as a project dataset so it can be trained on.</summary>
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> RegisterDisk(string name, int projectId = 1)
        {
            var safeName = string.Join("_", (name ?? string.Empty).Split(Path.GetInvalidFileNameChars()));
            var dir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "datasets", safeName);
            if (string.IsNullOrWhiteSpace(safeName) || !Directory.Exists(dir))
                return NotFound();

            var datasets = await _datasetService.GetDatasetsForProjectAsync(projectId);
            var dataset = datasets.FirstOrDefault(d =>
                string.Equals(d.StoragePath, dir, StringComparison.OrdinalIgnoreCase));

            if (dataset == null)
            {
                var total = CountImages(dir);
                var trainDir = Path.Combine(dir, "images", "train");
                var trainCount = Directory.Exists(trainDir) ? CountImages(trainDir) : 0;
                var valDir = Path.Combine(dir, "images", "val");
                var testDir = Path.Combine(dir, "images", "test");

                // Read the class count from data.yaml when present.
                var nc = 1;
                try
                {
                    foreach (var line in System.IO.File.ReadLines(Path.Combine(dir, "data.yaml")))
                    {
                        if (line.StartsWith("nc:"))
                        {
                            nc = int.TryParse(line.Substring(3).Trim(), out var v) ? Math.Max(1, v) : 1;
                            break;
                        }
                    }
                }
                catch { /* keep default */ }

                dataset = new Dataset
                {
                    ProjectId = projectId,
                    Name = safeName,
                    Format = DatasetFormat.YOLO_TXT,
                    StoragePath = dir,
                    TotalImages = total,
                    TotalLabeled = total,
                    Classes = System.Text.Json.JsonSerializer.Serialize(
                        Enumerable.Range(0, nc).Select(i => $"class_{i}").ToList()),
                    TrainCount = trainCount,
                    ValCount = Directory.Exists(valDir) ? CountImages(valDir) : 0,
                    TestCount = Directory.Exists(testDir) ? CountImages(testDir) : 0,
                    Status = DatasetStatus.Ready
                };
                await _datasetService.CreateDatasetAsync(dataset);

                TempData["SuccessMessage"] = $"'{safeName}' registered ({total} images). Ready to train!";
            }

            return RedirectToAction("Wizard", "Training", new { projectId, datasetId = dataset.Id });
        }

        /// <summary>Download the raw sample-dataset ZIP that is cached locally
        /// in wwwroot/datasets (the same archive used for installation).</summary>
        // Reached via the default route: /Dataset/DownloadSample/{id}
        public IActionResult DownloadSample(string id)
        {
            var key = (id ?? string.Empty).ToLowerInvariant();
            var safeKeys = new HashSet<string> { "coco8", "coco128", "dota8" };
            if (!safeKeys.Contains(key))
                return NotFound();

            var zipPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "datasets", $"{key}.zip");
            if (!System.IO.File.Exists(zipPath))
                return NotFound(); // not downloaded yet — install it first from the Datasets page

            return PhysicalFile(zipPath, "application/zip", $"{key}.zip");
        }
    }
}
