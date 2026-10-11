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
        public async Task<IActionResult> Stop(int id,
            [FromServices] YOLOVisionStudio.Data.ApplicationDbContext db)
        {
            var job = await db.TrainingJobs.FindAsync(id);
            if (job == null) return NotFound();
            if (job.Status == JobStatus.Running)
            {
                job.Status = JobStatus.Cancelled;
                job.ErrorMessage = "Manually stopped by user.";
                job.CompletedAt = DateTime.UtcNow;
                await db.SaveChangesAsync();
            }
            return Ok(new { success = true, status = job.Status.ToString() });
        }

        /// <summary>Return all training jobs so the 3D visualization pages can populate job selectors.</summary>
        [HttpGet("jobs")]
        public async Task<IActionResult> GetJobs()
        {
            var jobs = await _trainingService.GetAllJobsAsync();
            var result = jobs.Select(j => new
            {
                id = j.Id,
                name = j.Name,
                status = j.Status.ToString(),
                pythonJobId = j.PythonJobId,
                currentEpoch = j.CurrentEpoch,
                totalEpochs = j.Epochs,
                baseModel = j.BaseModel,
                yoloVersion = j.YoloVersion,
                bestMap50 = j.BestmAP50
            });
            return Ok(result);
        }

        /// <summary>Return a single job's current status and metrics.</summary>
        [HttpGet("{id}/status")]
        public async Task<IActionResult> GetStatus(int id)
        {
            var job = await _trainingService.GetJobByIdAsync(id);
            if (job == null) return NotFound();
            return Ok(new
            {
                id = job.Id,
                status = job.Status.ToString(),
                pythonJobId = job.PythonJobId,
                currentEpoch = job.CurrentEpoch,
                totalEpochs = job.Epochs,
                bestMap50 = job.BestmAP50,
                bestMap5095 = job.BestmAP5095,
                errorMessage = job.ErrorMessage
            });
        }

        /// <summary>Mark all stuck Running jobs (Python engine lost them) as Failed immediately.</summary>
        [HttpPost("reset-stale")]
        public async Task<IActionResult> ResetStale([FromServices] YOLOVisionStudio.Data.ApplicationDbContext db,
                                                     [FromServices] IYoloAiService ai)
        {
            var running = db.TrainingJobs
                .Where(j => j.Status == JobStatus.Running)
                .ToList();

            int reset = 0;
            foreach (var job in running)
            {
                if (string.IsNullOrEmpty(job.PythonJobId)) continue;
                try
                {
                    var m = await ai.GetTrainingMetrics(job.PythonJobId);
                    if (string.IsNullOrEmpty(m.Status))
                    {
                        job.Status = JobStatus.Failed;
                        job.ErrorMessage = "Stale: AI engine restarted. Please start a new run.";
                        reset++;
                    }
                }
                catch { /* engine offline */ }
            }
            await db.SaveChangesAsync();
            return Ok(new { reset, message = $"{reset} stale job(s) marked as Failed." });
        }

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
