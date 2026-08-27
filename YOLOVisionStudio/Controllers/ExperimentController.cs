using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class ExperimentController : Controller
    {
        public IActionResult Index() => View();
        public IActionResult Detail(int id) => View();
    }
}
