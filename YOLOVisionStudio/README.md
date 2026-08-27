# 🎯 YOLO Vision Studio

**A complete desktop studio for training, visualizing, and deploying YOLO computer-vision models — no command line required.**

YOLO Vision Studio wraps the full Ultralytics YOLO workflow in a polished ASP.NET Core web application backed by a FastAPI AI engine. Create projects, import datasets, split them into the YOLO folder structure automatically, launch training with live metrics streaming, run inference on images or your webcam, compare experiments, export models — all from one interface built for beginners and practitioners alike.

---

## ✨ Features

### 📁 Project & Dataset Management
- **Project workspace** — organize datasets, models, and training runs per project.
- **One-click sample datasets** — install COCO8, COCO128, or DOTA8 directly from Ultralytics assets.
- **Smart dataset upload** — drop a `.zip` with *any* internal structure; the studio extracts it and inspects it automatically.
- **Automatic YOLO structuring** — if a dataset doesn't follow `images/{train,val,test}` + `labels/...` + `data.yaml`, it's split into that structure in one click with user-defined ratios (default **70 / 15 / 15**).
- **Disk scanner** — every dataset folder under `wwwroot/datasets` is detected, checked for YOLO readiness, and offered for splitting or registration.

### 🏋️ Training
- **Staged wizard** — Dataset → Model → Hyperparameters → Train.
- **Full control** — YOLO version/task, base model scale, epochs, batch size, image size, optimizer, learning rate, weight decay, mosaic/mixup augmentation.
- **Live monitoring** — real-time epoch progress, loss curves, and metrics streamed over **SignalR**.
- **Background job worker** — training keeps running server-side; close the browser and come back.

### 🔬 Visualization (Explainability)
- **End-to-end architecture explorer** — the *real* model loaded from its weights: every layer including nested blocks (Conv → Conv2d + BatchNorm + SiLU, SPPF → MaxPool2d×3, Detect heads), with actual input/output tensor shapes captured during a live forward pass.
- **Beginner-friendly mode** — plain-English explanation for each operation type, color-coded roles, and a data-flow strip showing how the image transforms step by step (`[1,3,640,640] → ... → boxes`).
- **Feature-map explorer** — hook intermediate layers and see what the network actually "sees" as heatmaps.
- **Prediction explainability pipeline** (XAI).

### 🎯 Inference
- Run predictions on single images or batches through the UI.
- **Webcam / camera stream** inference in the browser.
- Model status checks and automatic weight download.

### 📊 Experiments & Export
- **Comparison dashboard** — evaluate multiple trained models side by side.
- **Experiment tracking** with per-run history stored in SQLite.
- **Export** models to deployment formats (ONNX, TorchScript, etc.).

### 🖥️ Operations
- **System monitor** — live CPU / RAM / GPU / disk usage of the machine powering your training.
- **Built-in assistant** chat panel.
- **Zero-config startup** — the .NET app auto-launches the Python engine; SQLite database is created and seeded on first run.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           YOLOVisionStudio (.NET)            │
│   ASP.NET Core MVC · SignalR · EF Core       │
│   UI · Projects · Datasets · Jobs · Export   │
└──────────────────┬──────────────────────────┘
                   │ REST (HttpClient)
┌──────────────────▼──────────────────────────┐
│         python-engine (FastAPI)              │
│   Ultralytics YOLO · PyTorch · OpenCV        │
│   Training · Inference · Splitting · XAI     │
└─────────────────────────────────────────────┘
```

| Layer | Technology |
|---|---|
| Frontend | Razor Views, vanilla JS, custom CSS design system |
| Backend | ASP.NET Core (.NET), SignalR real-time hub |
| Data | Entity Framework Core + SQLite (auto-created) |
| AI Engine | Python 3.10+, FastAPI, Uvicorn |
| AI Stack | Ultralytics YOLO, PyTorch, OpenCV |

---

## 🚀 Getting Started

### Prerequisites
- [.NET SDK](https://dotnet.microsoft.com/download) (net10.0)
- [Python 3.10+](https://www.python.org/) with an environment containing:
  `ultralytics`, `torch`, `torchvision`, `opencv-python`, `tensorboard`
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Setup

1. **Clone**
   ```bash
   git clone https://github.com/<you>/YOLOVisionStudio.git
   cd YOLOVisionStudio
   ```

2. **Python engine**
   ```bash
   cd python-engine
   pip install -r requirements.txt
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   Or simply run `start.bat`.

3. **Web application**
   ```bash
   cd YOLOVisionStudio
   dotnet run
   ```

4. Open `https://localhost:7135` — the app launches/reuses the Python engine automatically and seeds the database on first run.

> The first training run downloads the selected base weights (e.g., `yolo11n.pt`) automatically.

---

## 📂 Project Structure

```
YOLOVisionStudio/
├── YOLOVisionStudio/            # ASP.NET Core web app
│   ├── Controllers/             # MVC + API controllers
│   ├── Services/                # AI-engine client, training worker, monitors
│   ├── Models/Domain|ViewModels # EF entities and view models
│   ├── Views/                   # Razor UI (Dataset, Training, Visualization…)
│   ├── Hubs/                    # SignalR real-time training updates
│   └── wwwroot/datasets/        # Local dataset storage
└── python-engine/               # FastAPI microservice
    ├── routers/                 # training, dataset, inference, visualization…
    ├── utils/                   # YOLO manager, image utilities
    └── main.py                  # App entry point
```

---

## ⚙️ Configuration

| Setting | Where | Default |
|---|---|---|
| Connection string | `appsettings.json` | `Data Source=YOLOVisionStudio.db` |
| AI engine URL | `YoloAiEngine:BaseUrl` | `http://localhost:8000` |
| Engine timeout | `YoloAiEngine:TimeoutSeconds` | `30` |
| Dataset directory | env `YOLO_DATASETS_DIR` | `wwwroot/datasets` |

---

## 🗺️ Roadmap

- [ ] Pure C# inference via ONNX Runtime (no Python needed for prediction)
- [ ] Dataset annotation tools
- [ ] Active-learning suggestions (label assistance)
- [ ] Multi-GPU training profiles

## 🤝 Contributing

Issues and pull requests are welcome — especially improvements to the visualization modules and C#/ONNX integration.

## 📄 License

This project is provided for educational and research use. YOLO models are powered by [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
