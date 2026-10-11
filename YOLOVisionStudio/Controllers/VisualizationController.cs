using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Threading.Tasks;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] — Local Studio Mode: no auth required
    public class VisualizationController : Controller
    {
        private readonly IYoloAiService _aiService;
        private readonly HttpClient _httpClient;

        public VisualizationController(IYoloAiService aiService, HttpClient httpClient)
        {
            _aiService = aiService;
            _httpClient = httpClient;
        }

        // Visualization Hub Index
        [HttpGet]
        public IActionResult Index()
        {
            ViewData["Title"] = "Visualization Hub";
            return View();
        }

        // Module 5: Neural Network Architecture Visualizer
        [HttpGet]
        public async Task<IActionResult> Architecture(string version = "v11", string size = "n")
        {
            try
            {
                var res = await _aiService.GetModelArchitecture(version, size);
                ViewData["Title"] = "Architecture Visualizer";
                return View(res);
            }
            catch
            {
                ViewData["Title"] = "Architecture Visualizer";
                return View(null);
            }
        }

        // Module 6: Feature Map Explorer
        [HttpGet]
        public IActionResult FeatureMaps()
        {
            ViewData["Title"] = "Feature Map Explorer";
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> FeatureMaps(string imagePath, string modelPath, string layerIndices)
        {
            var layers = new System.Collections.Generic.List<int>();
            if (!string.IsNullOrEmpty(layerIndices))
            {
                foreach (var s in layerIndices.Split(','))
                    if (int.TryParse(s.Trim(), out int idx)) layers.Add(idx);
            }
            if (layers.Count == 0) layers = new System.Collections.Generic.List<int> { 0, 2, 4, 8 };

            try
            {
                var res = await _aiService.GetFeatureMaps(new FeatureMapsRequest
                {
                    ImagePath = imagePath,
                    ModelPath = modelPath,
                    LayerIndices = layers
                });
                ViewData["Title"] = "Feature Map Explorer";
                return View("FeatureMapsResult", res);
            }
            catch
            {
                ViewData["Title"] = "Feature Map Explorer";
                return View();
            }
        }

        // Module 7: Real Prediction Explainability (XAI)
        [HttpGet]
        public IActionResult Explainability()
        {
            ViewData["Title"] = "Prediction Explainability";
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> GenerateGradCAM([FromBody] GradCAMRequest request)
        {
            try
            {
                var res = await _aiService.GenerateGradCAM(request);
                return Json(res);
            }
            catch (Exception ex)
            {
                return Json(new GradCAMResponse { OriginalImageBase64 = "", HeatmapBase64 = "", OverlayBase64 = "" });
            }
        }

        [HttpPost]
        public async Task<IActionResult> GenerateAttentionMaps([FromBody] AttentionMapRequest request)
        {
            try
            {
                var res = await _aiService.GenerateAttentionMaps(request);
                return Json(res);
            }
            catch (Exception ex)
            {
                return Json(new AttentionMapResponse { OriginalImageBase64 = "", AttentionMaps = new(), OverlayBase64 = "" });
            }
        }

        [HttpGet]
        public async Task<IActionResult> VisualizeWeights(string modelPath, int layerIndex = 0)
        {
            try
            {
                var res = await _aiService.VisualizeWeights(modelPath, layerIndex);
                return Json(res);
            }
            catch (Exception ex)
            {
                return Json(new WeightVisualization { LayerIndex = layerIndex, LayerName = "Error", WeightsBase64 = "" });
            }
        }

        [HttpGet]
        public IActionResult Weights()
        {
            ViewData["Title"] = "Weight & Filter Visualization";
            return View();
        }

        [HttpGet]
        public IActionResult LayerStats()
        {
            ViewData["Title"] = "Layer Statistics Dashboard";
            return View();
        }

        [HttpGet]
        public async Task<IActionResult> GetLayerStatistics(string jobId)
        {
            try
            {
                var response = await _httpClient.GetAsync($"/visualization/layer-statistics/{jobId}");
                var data = await response.Content.ReadAsStringAsync();
                return Content(data, "application/json");
            }
            catch (Exception ex)
            {
                return Json(new { job_id = jobId, layer_statistics = new List<object>(), message = "Failed to load statistics" });
            }
        }

        [HttpGet]
        public IActionResult Network3D()
        {
            ViewData["Title"] = "3D Network Topology";
            return View();
        }

        [HttpGet]
        public IActionResult EpochComparison()
        {
            ViewData["Title"] = "Epoch Activation Comparison";
            return View();
        }

        [HttpGet]
        public IActionResult LiveTraining3D()
        {
            ViewData["Title"] = "Live 3D Training Visualization";
            return View();
        }

        [HttpGet]
        public IActionResult NetworkMonitor3D()
        {
            ViewData["Title"] = "3D Network Monitor";
            return View();
        }

        /// <summary>JSON-only architecture endpoint for client-side 3D viz JS.</summary>
        [HttpGet]
        public async Task<IActionResult> GetModelArchitectureJson(string yoloVersion = "v11", string modelSize = "n")
        {
            try
            {
                var res = await _aiService.GetModelArchitecture(yoloVersion, modelSize);
                return Json(res);
            }
            catch
            {
                return Json(new { layers = new List<object>(), total_parameters = 0, total_flops = 0 });
            }
        }

        /// <summary>Full-screen CNN stacked-planes 3D live visualization — real activation maps per channel.</summary>
        [HttpGet]
        public IActionResult CNNViz3D()
        {
            ViewData["Title"] = "CNN 3D Live Visualization";
            return View();
        }

    }
}

