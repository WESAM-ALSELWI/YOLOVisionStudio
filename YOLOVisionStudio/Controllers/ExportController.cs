using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class ExportController : Controller
    {
        private readonly IYoloAiService _aiService;
        public ExportController(IYoloAiService aiService) => _aiService = aiService;

        public IActionResult Index(int modelId)
        {
            ViewBag.ModelId = modelId;
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Export(int modelId, string format)
        {
            var res = await _aiService.ExportModel(new ExportRequest { ModelId = modelId, Format = format });
            return RedirectToAction("Index", new { modelId });
        }
    }
}
