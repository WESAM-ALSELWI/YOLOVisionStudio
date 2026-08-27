using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;
using YOLOVisionStudio.Services;

namespace YOLOVisionStudio.Controllers.Api
{
    [Route("api/[controller]")]
    [ApiController]
    // [Authorize] (Local Mode Enabled)
    public class ApiProjectController : ControllerBase
    {
        private readonly IProjectService _projectService;
        public ApiProjectController(IProjectService projectService) => _projectService = projectService;

        [HttpGet]
        public async Task<IActionResult> GetProjects() => Ok(await _projectService.GetAllProjectsAsync());

        [HttpGet("{id}")]
        public async Task<IActionResult> GetProject(int id)
        {
            var project = await _projectService.GetProjectByIdAsync(id);
            if (project == null) return NotFound();
            return Ok(project);
        }

        [HttpPost]
        public async Task<IActionResult> CreateProject([FromBody] Project project)
        {
            var created = await _projectService.CreateProjectAsync(project);
            return CreatedAtAction(nameof(GetProject), new { id = created.Id }, created);
        }

        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteProject(int id)
        {
            await _projectService.DeleteProjectAsync(id);
            return NoContent();
        }
    }

    [Route("api/datasets")]
    [ApiController]
    // [Authorize] (Local Mode Enabled)
    public class ApiDatasetController : ControllerBase
    {
        private readonly IDatasetService _datasetService;
        public ApiDatasetController(IDatasetService datasetService) => _datasetService = datasetService;

        [HttpGet]
        public async Task<IActionResult> GetDatasets(int projectId) => Ok(await _datasetService.GetDatasetsForProjectAsync(projectId));
    }

    [Route("api/training")]
    [ApiController]
    // [Authorize] (Local Mode Enabled)
    public class ApiTrainingController : ControllerBase
    {
        private readonly ITrainingService _trainingService;
        public ApiTrainingController(ITrainingService trainingService) => _trainingService = trainingService;

        [HttpPost("start")]
        public async Task<IActionResult> Start([FromBody] TrainingJob job) => Ok(await _trainingService.StartJobAsync(job));

        [HttpPost("{id}/stop")]
        public IActionResult Stop(int id) => Ok(new { success = true });
    }

    [Route("api/models")]
    [ApiController]
    public class ApiModelsController : ControllerBase
    {
        private readonly IYoloAiService _aiService;
        public ApiModelsController(IYoloAiService aiService) => _aiService = aiService;

        /// <summary>Check whether a pretrained model already exists in wwwroot/models.</summary>
        [HttpGet("status")]
        public async Task<IActionResult> Status([FromQuery] string name) => Ok(await _aiService.GetModelStatus(name));

        /// <summary>Download the model into wwwroot/models only if it is not cached yet.</summary>
        [HttpPost("ensure")]
        public async Task<IActionResult> Ensure([FromBody] ModelsEnsureBody body)
            => Ok(await _aiService.EnsureModel(body.Model ?? string.Empty));
    }

    public class ModelsEnsureBody
    {
        public string? Model { get; set; }
    }
}
