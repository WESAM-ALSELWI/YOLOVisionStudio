using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using YOLOVisionStudio.Data;
using YOLOVisionStudio.Hubs;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Services
{
    public class TrainingBackgroundWorker : BackgroundService
    {
        private readonly IServiceScopeFactory _scopeFactory;

        public TrainingBackgroundWorker(IServiceScopeFactory scopeFactory)
        {
            _scopeFactory = scopeFactory;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                await Task.Delay(2000, stoppingToken);

                try
                {
                    using var scope = _scopeFactory.CreateScope();
                    var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
                    var aiService = scope.ServiceProvider.GetRequiredService<IYoloAiService>();
                    var hubContext = scope.ServiceProvider.GetRequiredService<IHubContext<TrainingHub>>();

                    var runningJobs = context.TrainingJobs.Where(j => j.Status == JobStatus.Running).ToList();

                    foreach (var job in runningJobs)
                    {
                        if (string.IsNullOrEmpty(job.PythonJobId)) continue;

                        TrainingMetricsResponse metrics;
                        try
                        {
                            metrics = await aiService.GetTrainingMetrics(job.PythonJobId);
                        }
                        catch
                        {
                            // Python engine unreachable — leave job alone for now, retry next cycle
                            continue;
                        }

                        // Python returned a 404 / empty response for this job — engine was restarted
                        // and the in-memory job state was lost.  Mark it as failed so it stops being polled.
                        if (string.IsNullOrEmpty(metrics.Status) && metrics.Epoch == 0 && !metrics.IsFinished)
                        {
                            job.Status = JobStatus.Failed;
                            job.ErrorMessage = "Training engine was restarted and lost this job's state. Please start a new training run.";
                            await context.SaveChangesAsync(stoppingToken);
                            await hubContext.Clients.Group($"job_{job.Id}").SendAsync("JobStatusChanged", job.Status.ToString(), cancellationToken: stoppingToken);
                            continue;
                        }

                        if (!string.IsNullOrEmpty(metrics.Status))
                        {
                            var pyStatus = metrics.Status;
                            if (pyStatus == "completed")
                                job.Status = JobStatus.Completed;
                            else if (pyStatus == "failed" || metrics.IsFailed)
                                job.Status = JobStatus.Failed;
                            else if (pyStatus == "cancelled")
                                job.Status = JobStatus.Cancelled;

                            if (job.Status != JobStatus.Running && !string.IsNullOrEmpty(metrics.ErrorMessage))
                                job.ErrorMessage = metrics.ErrorMessage;
                        }

                        bool changed = false;
                        if (metrics.Epoch > job.CurrentEpoch)
                        {
                            job.CurrentEpoch = metrics.Epoch;
                            changed = true;
                        }
                        if (metrics.mAP50 > job.BestmAP50) { job.BestmAP50 = metrics.mAP50; changed = true; }
                        if (metrics.mAP5095 > job.BestmAP5095) { job.BestmAP5095 = metrics.mAP5095; changed = true; }

                        if (changed || job.Status != JobStatus.Running)
                        {
                            var metric = new TrainingMetric
                            {
                                TrainingJobId = job.Id,
                                Epoch = metrics.Epoch,
                                mAP50 = metrics.mAP50,
                                mAP5095 = metrics.mAP5095,
                                Timestamp = DateTime.UtcNow
                            };
                            context.TrainingMetrics.Add(metric);
                            await context.SaveChangesAsync(stoppingToken);

                            await hubContext.Clients.Group($"job_{job.Id}").SendAsync("ReceiveMetrics", metric, cancellationToken: stoppingToken);
                            await hubContext.Clients.Group($"job_{job.Id}").SendAsync("JobStatusChanged", job.Status.ToString(), cancellationToken: stoppingToken);
                        }
                    }

                }
                catch
                {
                    // Ignore DB initialization window exceptions
                }
            }
        }
    }
}
