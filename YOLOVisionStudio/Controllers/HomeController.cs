using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.ViewModels;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers
{
    public class HomeController : Controller
    {
        private readonly ISystemMonitorService _monitorService;
        private readonly IProjectService _projectService;

        public HomeController(ISystemMonitorService monitorService, IProjectService projectService)
        {
            _monitorService = monitorService;
            _projectService = projectService;
        }

        public async Task<IActionResult> Index()
        {
            var stats = await _monitorService.GetStatsAsync();
            var projects = await _projectService.GetAllProjectsAsync();
            
            var vm = new DashboardViewModel
            {
                ProjectCount = projects.Count,
                RecentProjects = projects,
                SystemStats = stats
            };
            return View(vm);
        }
    }
}
