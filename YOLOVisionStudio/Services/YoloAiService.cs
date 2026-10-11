using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;

namespace YOLOVisionStudio.Services
{
    public class YoloAiService : IYoloAiService
    {
        private readonly HttpClient _httpClient;

        public YoloAiService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<TrainingStartResponse> StartTraining(TrainingRequest request)
        {
            try {
                var response = await _httpClient.PostAsJsonAsync("/training/start", request);
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<TrainingStartResponse>() ?? new TrainingStartResponse { Success = false };
            } catch (Exception ex) {
                return new TrainingStartResponse { Success = false, Message = ex.Message };
            }
        }

        public async Task<TrainingMetricsResponse> GetTrainingMetrics(string jobId)
        {
            try {
                var response = await _httpClient.GetAsync($"/training/{jobId}/metrics");
                if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                    return new TrainingMetricsResponse(); // empty — job not found in Python engine
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<TrainingMetricsResponse>() ?? new TrainingMetricsResponse();
            } catch (HttpRequestException) {
                throw; // let background worker handle connection failures
            } catch { return new TrainingMetricsResponse(); }
        }

        public async Task<StopTrainingResponse> StopTraining(string jobId)
        {
            try {
                var response = await _httpClient.PostAsync($"/training/{jobId}/stop", null);
                return new StopTrainingResponse { Success = response.IsSuccessStatusCode };
            } catch { return new StopTrainingResponse { Success = false }; }
        }

        public async Task<ModelStatusResponse> GetModelStatus(string modelName)
        {
            try {
                return await _httpClient.GetFromJsonAsync<ModelStatusResponse>($"/models/status/{Uri.EscapeDataString(modelName)}") ?? new ModelStatusResponse();
            } catch { return new ModelStatusResponse(); }
        }

        public async Task<EnsureModelResponse> EnsureModel(string modelName)
        {
            try {
                var response = await _httpClient.PostAsJsonAsync("/models/ensure", new { model = modelName });
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<EnsureModelResponse>() ?? new EnsureModelResponse();
            } catch { return new EnsureModelResponse(); }
        }

        public async Task<List<CachedModelInfo>> ListCachedModels()
        {
            try
            {
                var result = await _httpClient.GetFromJsonAsync<CachedModelsEnvelope>("/models");
                return result?.Models ?? new List<CachedModelInfo>();
            }
            catch { return new List<CachedModelInfo>(); }
        }

        public async Task<InstallSampleResponse> InstallSampleDataset(string key)
        {
            try {
                var response = await _httpClient.PostAsJsonAsync("/dataset/install-sample", new InstallSampleRequest { Name = key });
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<InstallSampleResponse>() ?? new InstallSampleResponse { Success = false };
            } catch (Exception ex) {
                return new InstallSampleResponse { Success = false, Message = ex.Message };
            }
        }

        public async Task<SplitDatasetResponse> SplitDataset(SplitDatasetRequest request)
        {
            try {
                var response = await _httpClient.PostAsJsonAsync("/dataset/split", request);
                if (!response.IsSuccessStatusCode)
                    return new SplitDatasetResponse { Success = false, Message = await response.Content.ReadAsStringAsync() };
                return await response.Content.ReadFromJsonAsync<SplitDatasetResponse>() ?? new SplitDatasetResponse { Success = false };
            } catch (Exception ex) {
                return new SplitDatasetResponse { Success = false, Message = ex.Message };
            }
        }

        public async Task<SplitStatusResponse> GetSplitStatus(string datasetPath)
        {
            try {
                return await _httpClient.GetFromJsonAsync<SplitStatusResponse>($"/dataset/split-status?path={Uri.EscapeDataString(datasetPath)}") ?? new SplitStatusResponse();
            } catch { return new SplitStatusResponse(); }
        }

        public async Task<InferenceResponse> RunInference(InferenceRequest request)
        {
            var res = await _httpClient.PostAsJsonAsync("/inference/image", request);
            return await res.Content.ReadFromJsonAsync<InferenceResponse>() ?? new InferenceResponse();
        }

        public async Task<DatasetAnalysisResponse> AnalyzeDataset(string datasetPath)
        {
            var res = await _httpClient.GetFromJsonAsync<DatasetAnalysisResponse>($"/dataset/analyze?path={Uri.EscapeDataString(datasetPath)}");
            return res ?? new DatasetAnalysisResponse();
        }

        public async Task<SystemStatsResponse> GetSystemStats()
        {
            try {
                return await _httpClient.GetFromJsonAsync<SystemStatsResponse>("/system/stats") ?? new SystemStatsResponse();
            } catch { return new SystemStatsResponse(); }
        }

        public async Task<FeatureMapsResponse> GetFeatureMaps(FeatureMapsRequest request)
        {
            var res = await _httpClient.PostAsJsonAsync("/visualization/feature-maps", request);
            return await res.Content.ReadFromJsonAsync<FeatureMapsResponse>() ?? new FeatureMapsResponse();
        }

        public async Task<ExportResponse> ExportModel(ExportRequest request)
        {
            var res = await _httpClient.PostAsJsonAsync("/export/model", request);
            return await res.Content.ReadFromJsonAsync<ExportResponse>() ?? new ExportResponse();
        }

        public async Task<ArchitectureResponse> GetModelArchitecture(string yoloVersion, string modelSize)
        {
            var res = await _httpClient.GetFromJsonAsync<ArchitectureResponse>($"/visualization/architecture/{yoloVersion}/{modelSize}");
            return res ?? new ArchitectureResponse();
        }

        public async Task<AugmentationPreviewResponse> GetAugmentationPreview(AugmentationRequest request)
        {
            var res = await _httpClient.PostAsJsonAsync("/dataset/augment", request);
            return await res.Content.ReadFromJsonAsync<AugmentationPreviewResponse>() ?? new AugmentationPreviewResponse();
        }

        public async Task<GradCAMResponse> GenerateGradCAM(GradCAMRequest request)
        {
            try
            {
                var res = await _httpClient.PostAsJsonAsync("/visualization/gradcam", request);
                res.EnsureSuccessStatusCode();
                return await res.Content.ReadFromJsonAsync<GradCAMResponse>() ?? new GradCAMResponse();
            }
            catch (Exception ex)
            {
                return new GradCAMResponse { OriginalImageBase64 = "", HeatmapBase64 = "", OverlayBase64 = "" };
            }
        }

        public async Task<AttentionMapResponse> GenerateAttentionMaps(AttentionMapRequest request)
        {
            try
            {
                var res = await _httpClient.PostAsJsonAsync("/visualization/attention-maps", request);
                res.EnsureSuccessStatusCode();
                return await res.Content.ReadFromJsonAsync<AttentionMapResponse>() ?? new AttentionMapResponse();
            }
            catch (Exception ex)
            {
                return new AttentionMapResponse { OriginalImageBase64 = "", AttentionMaps = new(), OverlayBase64 = "" };
            }
        }

        public async Task<WeightVisualization> VisualizeWeights(string modelPath, int layerIndex)
        {
            try
            {
                var res = await _httpClient.GetAsync($"/visualization/weights/{Uri.EscapeDataString(modelPath)}?layer_index={layerIndex}");
                res.EnsureSuccessStatusCode();
                return await res.Content.ReadFromJsonAsync<WeightVisualization>() ?? new WeightVisualization();
            }
            catch (Exception ex)
            {
                return new WeightVisualization { LayerIndex = layerIndex, LayerName = "Error", WeightsBase64 = "" };
            }
        }

        private class CachedModelsEnvelope
        {
            [System.Text.Json.Serialization.JsonPropertyName("models")]
            public List<CachedModelInfo>? Models { get; set; }
        }
    }
}
