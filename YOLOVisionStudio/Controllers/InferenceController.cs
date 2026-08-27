using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class InferenceController : Controller
    {
        public IActionResult Test() => View();
        public IActionResult Camera() => View();
    }
}
