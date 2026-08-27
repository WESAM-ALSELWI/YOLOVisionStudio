using System.Collections.Generic;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public interface IProjectService
    {
        Task<List<Project>> GetAllProjectsAsync();
        Task<Project?> GetProjectByIdAsync(int id);
        Task<Project> CreateProjectAsync(Project project);
        Task UpdateProjectAsync(Project project);
        Task DeleteProjectAsync(int id);
    }
}
