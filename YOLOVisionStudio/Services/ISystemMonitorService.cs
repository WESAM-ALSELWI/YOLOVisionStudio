using System.Threading.Tasks;
using YOLOVisionStudio.Models.ViewModels;

namespace YOLOVisionStudio.Services
{
    public interface ISystemMonitorService
    {
        Task<MonitorStats> GetStatsAsync();
    }
}
