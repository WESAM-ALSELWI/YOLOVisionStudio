using System.Threading.Tasks;
using YOLOVisionStudio.Models.ViewModels;

namespace YOLOVisionStudio.Services
{
    public class SystemMonitorService : ISystemMonitorService
    {
        private readonly IYoloAiService _aiService;

        public SystemMonitorService(IYoloAiService aiService)
        {
            _aiService = aiService;
        }

        public async Task<MonitorStats> GetStatsAsync()
        {
            var gpuStats = await _aiService.GetSystemStats();
            return new MonitorStats
            {
                CpuUsagePercentage = 15.2, // Simulated for now
                RamUsageMB = 4096,
                TotalRamMB = 16384,
                GpuUsagePercentage = gpuStats.GpuUsage,
                GpuName = gpuStats.GpuName
            };
        }
    }
}
