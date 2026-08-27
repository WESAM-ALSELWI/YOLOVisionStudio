using System.Collections.Generic;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public interface ITrainingService
    {
        Task<TrainingJob> StartJobAsync(TrainingJob job);
        Task<TrainingJob?> GetJobByIdAsync(int id);
        Task<List<TrainingJob>> GetAllJobsAsync();
    }
}
