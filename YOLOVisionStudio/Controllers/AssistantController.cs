using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using YOLOVisionStudio.Models.ViewModels;
using System.Collections.Generic;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class AssistantController : Controller
    {
        public IActionResult Index()
        {
            return View(new AssistantViewModel());
        }

        [HttpPost]
        public IActionResult Chat([FromBody] ChatMessage message)
        {
            string reply = "I recommend YOLOv8 medium for general tasks.";
            if (message.Content.ToLower().Contains("fast") || message.Content.ToLower().Contains("edge"))
                reply = "For fast edge inference, YOLOv8 nano or YOLOv10 nano is recommended.";
            if (message.Content.ToLower().Contains("segment"))
                reply = "You should use the Segmentation objective and a YOLOv8-seg model.";
                
            return Json(new { reply });
        }
    }
}
