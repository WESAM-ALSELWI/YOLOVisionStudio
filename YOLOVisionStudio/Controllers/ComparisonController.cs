using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class ComparisonController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}
