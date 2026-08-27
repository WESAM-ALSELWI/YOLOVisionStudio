using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using YOLOVisionStudio.Data;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public class ProjectService : IProjectService
    {
        private readonly ApplicationDbContext _context;
        public ProjectService(ApplicationDbContext context) => _context = context;

        public async Task<List<Project>> GetAllProjectsAsync()
        {
            return await _context.Projects.Include(p => p.CreatedByUser).ToListAsync();
        }

        public async Task<Project?> GetProjectByIdAsync(int id)
        {
            return await _context.Projects.Include(p => p.Datasets).Include(p => p.Models).Include(p => p.Experiments).FirstOrDefaultAsync(p => p.Id == id);
        }

        public async Task<Project> CreateProjectAsync(Project project)
        {
            _context.Projects.Add(project);
            await _context.SaveChangesAsync();
            return project;
        }

        public async Task UpdateProjectAsync(Project project)
        {
            _context.Projects.Update(project);
            await _context.SaveChangesAsync();
        }

        public async Task DeleteProjectAsync(int id)
        {
            var project = await _context.Projects.FindAsync(id);
            if (project != null)
            {
                _context.Projects.Remove(project);
                await _context.SaveChangesAsync();
            }
        }
    }
}
