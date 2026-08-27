using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;
using YOLOVisionStudio.Models.ViewModels;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers
{
    public class ProjectController : Controller
    {
        private readonly IProjectService _projectService;

        public ProjectController(IProjectService projectService)
        {
            _projectService = projectService;
        }

        public async Task<IActionResult> Index()
        {
            var projects = await _projectService.GetAllProjectsAsync();
            return View(projects);
        }

        public async Task<IActionResult> Detail(int id)
        {
            var project = await _projectService.GetProjectByIdAsync(id);
            if (project == null) return NotFound();
            return View(new ProjectViewModel { Project = project });
        }

        [HttpGet]
        public IActionResult Create() => View(new ProjectCreateViewModel());

        [HttpPost]
        public async Task<IActionResult> Create(ProjectCreateViewModel model)
        {
            if (ModelState.IsValid)
            {
                var project = new Project
                {
                    Name = model.Name,
                    Description = model.Description,
                    ObjectiveType = model.ObjectiveType,
                    CreatedByUserId = "user-admin"
                };
                await _projectService.CreateProjectAsync(project);
                return RedirectToAction("Detail", new { id = project.Id });
            }
            return View(model);
        }

        [HttpPost]
        public async Task<IActionResult> Delete(int id)
        {
            await _projectService.DeleteProjectAsync(id);
            return RedirectToAction("Index");
        }
    }
}
