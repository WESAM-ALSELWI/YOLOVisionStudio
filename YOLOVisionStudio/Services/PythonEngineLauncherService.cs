using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace YOLOVisionStudio.Services
{
    public class PythonEngineLauncherService : IHostedService
    {
        private readonly ILogger<PythonEngineLauncherService> _logger;
        private readonly IConfiguration _config;
        private Process? _pythonProcess;

        public PythonEngineLauncherService(ILogger<PythonEngineLauncherService> logger, IConfiguration config)
        {
            _logger = logger;
            _config = config;
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            var baseUrl = _config["YoloAiEngine:BaseUrl"] ?? "http://localhost:8000";
            var condaEnvPath = _config["YoloAiEngine:CondaEnvPath"] ?? @"D:\WorkSpace\CondaEnv\AmalAI";
            var modelsPath = _config["YoloAiEngine:ModelsPath"] ?? "wwwroot/models";

            // Check if Python engine is already running
            bool isRunning = await IsEngineRunningAsync(baseUrl);
            if (isRunning)
            {
                _logger.LogInformation("Python AI Engine is already running at {BaseUrl}", baseUrl);
                return;
            }

            var pythonExe = Path.Combine(condaEnvPath, "python.exe");
            var engineDir = Path.Combine(Directory.GetCurrentDirectory(), "python-engine");
            if (!Directory.Exists(engineDir))
            {
                engineDir = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "python-engine"));
            }

            if (!File.Exists(pythonExe))
            {
                _logger.LogWarning("Python executable not found at {PythonExe}. Please start Python engine manually.", pythonExe);
                return;
            }

            // Resolve shared directories inside wwwroot
            var modelsDir = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), modelsPath));
            Directory.CreateDirectory(modelsDir);
            var datasetsDir = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "datasets"));
            Directory.CreateDirectory(datasetsDir);
            _logger.LogInformation("📁 YOLO models dir: {ModelsDir} | datasets dir: {DatasetsDir}", modelsDir, datasetsDir);

            _logger.LogInformation("🚀 Automatically launching Python AI Engine using {PythonExe} from {EngineDir}...", pythonExe, engineDir);

            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "-m uvicorn main:app --host 0.0.0.0 --port 8000",
                    WorkingDirectory = engineDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false
                };

                // Pass shared storage locations to the Python engine
                startInfo.EnvironmentVariables["YOLO_MODELS_DIR"] = modelsDir;
                startInfo.EnvironmentVariables["YOLO_DATASETS_DIR"] = datasetsDir;

                _pythonProcess = Process.Start(startInfo);
                _logger.LogInformation("Python AI Engine started (PID: {PID}).", _pythonProcess?.Id);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to automatically launch Python AI Engine.");
            }
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                try
                {
                    _logger.LogInformation("Stopping Python AI Engine (PID: {PID})...", _pythonProcess.Id);
                    _pythonProcess.Kill(entireProcessTree: true);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error shutting down Python process.");
                }
            }
            return Task.CompletedTask;
        }

        private static async Task<bool> IsEngineRunningAsync(string baseUrl)
        {
            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
                var response = await client.GetAsync($"{baseUrl}/health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }
    }
}
