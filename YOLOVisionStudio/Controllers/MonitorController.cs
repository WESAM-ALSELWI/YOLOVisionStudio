using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers
{
    // [Authorize] (Local Mode Enabled)
    public class MonitorController : Controller
    {
        private readonly ISystemMonitorService _monitorService;

        public MonitorController(ISystemMonitorService monitorService)
        {
            _monitorService = monitorService;
        }

        public IActionResult Index() => View();

        [HttpGet]
        public async Task<IActionResult> Stats()
        {
            var stats = await _monitorService.GetStatsAsync();
            return Json(stats);
        }
    }
}
