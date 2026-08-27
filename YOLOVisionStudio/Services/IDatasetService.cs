using System.Collections.Generic;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public interface IDatasetService
    {
        Task<List<Dataset>> GetDatasetsForProjectAsync(int projectId);
        Task<Dataset> CreateDatasetAsync(Dataset dataset);
        Task<Dataset> UpdateDatasetAsync(Dataset dataset);
        Task<Dataset?> GetDatasetByIdAsync(int id);
        Task DeleteDatasetAsync(int id);
    }
}
