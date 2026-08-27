using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using YOLOVisionStudio.Data;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public class DatasetService : IDatasetService
    {
        private readonly ApplicationDbContext _context;
        public DatasetService(ApplicationDbContext context) => _context = context;

        public async Task<List<Dataset>> GetDatasetsForProjectAsync(int projectId)
        {
            return await _context.Datasets.Where(d => d.ProjectId == projectId).ToListAsync();
        }

        public async Task<Dataset> CreateDatasetAsync(Dataset dataset)
        {
            _context.Datasets.Add(dataset);
            await _context.SaveChangesAsync();
            return dataset;
        }

        public async Task<Dataset> UpdateDatasetAsync(Dataset dataset)
        {
            _context.Datasets.Update(dataset);
            await _context.SaveChangesAsync();
            return dataset;
        }

        public async Task<Dataset?> GetDatasetByIdAsync(int id)
        {
            return await _context.Datasets.Include(d => d.Images).FirstOrDefaultAsync(d => d.Id == id);
        }

        public async Task DeleteDatasetAsync(int id)
        {
            var ds = await _context.Datasets.FindAsync(id);
            if (ds != null)
            {
                _context.Datasets.Remove(ds);
                await _context.SaveChangesAsync();
            }
        }
    }
}
