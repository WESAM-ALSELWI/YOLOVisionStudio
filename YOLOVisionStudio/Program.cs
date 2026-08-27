using Microsoft.AspNetCore.Builder;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using YOLOVisionStudio.Data;
using YOLOVisionStudio.Hubs;
using YOLOVisionStudio.Services;

var builder = WebApplication.CreateBuilder(args);

// Database
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection") ?? "Data Source=YOLOVisionStudio.db";
builder.Services.AddDbContext<ApplicationDbContext>(options => options.UseSqlite(connectionString));

// Services
var yoloBaseUrl = builder.Configuration["YoloAiEngine:BaseUrl"] ?? "http://localhost:8000";
builder.Services.AddHttpClient<IYoloAiService, YoloAiService>(client =>
{
    client.BaseAddress = new Uri(yoloBaseUrl);
    client.Timeout = TimeSpan.FromSeconds(builder.Configuration.GetValue<int>("YoloAiEngine:TimeoutSeconds", 30));
});

builder.Services.AddScoped<IProjectService, ProjectService>();
builder.Services.AddScoped<IDatasetService, DatasetService>();
builder.Services.AddScoped<ITrainingService, TrainingService>();
builder.Services.AddScoped<ISystemMonitorService, SystemMonitorService>();

// Hosted Services (Automatic Python AI Engine Launcher & Training Worker)
builder.Services.AddHostedService<PythonEngineLauncherService>();
builder.Services.AddHostedService<TrainingBackgroundWorker>();

// SignalR
builder.Services.AddSignalR();

// MVC and API
builder.Services.AddControllersWithViews();
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowPythonEngine",
        policy => policy.WithOrigins(yoloBaseUrl).AllowAnyMethod().AllowAnyHeader());
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}
else
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseCors("AllowPythonEngine");

app.MapHub<TrainingHub>("/hubs/training");
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");
app.MapControllers();

// Apply database creation & seeding
using (var scope = app.Services.CreateScope())
{
    var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
    context.Database.EnsureCreated();
}

app.Run();
