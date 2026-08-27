import os
import glob
from typing import List, Dict, Any

def get_yolo_model_info(version: str, size: str) -> str:
    """Return model filename based on version and size (e.g., yolo11n.pt, yolov8s.pt)."""
    return f"yolo{version.replace('v', '')}{size}.pt"

def get_available_tasks(version: str) -> List[str]:
    """Return supported tasks for a YOLO version."""
    # Simplified assumptions
    return ["detect", "segment", "classify", "pose", "obb"]

def parse_results_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Parse YOLO training results CSV into list of metric dicts."""
    if not os.path.exists(csv_path):
        return []
        
    results = []
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return []
                
            headers = [h.strip() for h in lines[0].split(',')]
            for line in lines[1:]:
                if not line.strip():
                    continue
                values = [v.strip() for v in line.split(',')]
                
                # Try to cast values to float
                row_dict = {}
                for h, v in zip(headers, values):
                    try:
                        row_dict[h] = float(v)
                    except ValueError:
                        row_dict[h] = v
                results.append(row_dict)
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        
    return results

def build_dataset_yaml(train_path: str, val_path: str, test_path: str, class_names: List[str], output_path: str) -> bool:
    """Write YOLO data.yaml file."""
    try:
        import yaml
    except ImportError:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml

    yaml_data = {
        'train': train_path,
        'val': val_path,
    }
    if test_path:
        yaml_data['test'] = test_path
        
    yaml_data['nc'] = len(class_names)
    yaml_data['names'] = class_names
    
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error creating dataset YAML: {e}")
        return False

def get_model_layer_graph(model_path: str) -> Dict[str, Any]:
    """Parse YOLO model architecture into nodes/edges."""
    # This is a stub for architecture parsing.
    # In reality, this requires loading model.model and inspecting PyTorch modules.
    from ultralytics import YOLO
    
    try:
        model = YOLO(model_path)
        # We simulate the structure here since fully automated deep introspection 
        # is complex and depends heavily on specific PyTorch versions.
        nodes = []
        
        for i, (name, module) in enumerate(model.model.named_modules()):
            if i == 0: continue # Skip root
            if "." in name: continue # Only take top-level blocks or refine as needed
            
            nodes.append({
                "id": i,
                "name": module.__class__.__name__,
                "type": module.__class__.__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
            })
            
        return {"nodes": nodes}
    except Exception as e:
        return {"error": str(e), "nodes": []}
