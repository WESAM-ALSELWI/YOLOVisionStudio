using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using YOLOVisionStudio.Services;
using YOLOVisionStudio.Models.ViewModels;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] — Local Studio Mode: no auth required
    public class VisualizationController : Controller
    {
        private readonly IYoloAiService _aiService;

        public VisualizationController(IYoloAiService aiService) => _aiService = aiService;

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

        // Module 7: Prediction Explainability (XAI Pipeline)
        [HttpGet]
        public IActionResult Explainability()
        {
            ViewData["Title"] = "Prediction Explainability";
            return View();
        }
    }
}

