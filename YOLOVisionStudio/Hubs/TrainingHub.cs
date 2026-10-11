using Microsoft.AspNetCore.SignalR;
using System.Threading.Tasks;
using YOLOVisionStudio.Models.Domain;

namespace YOLOVisionStudio.Hubs
{
    public class TrainingHub : Hub
    {
        public async Task JoinJobGroup(string jobId)
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, jobId);
        }

        public async Task LeaveJobGroup(string jobId)
        {
            await Groups.RemoveFromGroupAsync(Context.ConnectionId, jobId);
        }

        public async Task SubscribeToJob(string jobId)
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, $"job_{jobId}");
        }

        public async Task UnsubscribeFromJob(string jobId)
        {
            await Groups.RemoveFromGroupAsync(Context.ConnectionId, $"job_{jobId}");
        }
    }
}
