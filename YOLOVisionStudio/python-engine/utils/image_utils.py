import base64
import io
import cv2
import numpy as np

def numpy_to_base64(arr: np.ndarray, format: str = 'JPEG') -> str:
    """Convert numpy image array to base64 string."""
    if arr is None or arr.size == 0:
        return ""
    success, encoded_image = cv2.imencode(f'.{format.lower()}', arr)
    if not success:
        raise ValueError("Failed to encode image")
    byte_data = encoded_image.tobytes()
    b64_str = base64.b64encode(byte_data).decode('utf-8')
    return b64_str

def base64_to_numpy(b64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    if not b64_str:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    
    # Remove prefix if present (e.g., data:image/jpeg;base64,)
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
        
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 string to image")
    return img

def apply_heatmap(feature_map_2d: np.ndarray) -> np.ndarray:
    """Normalize and apply VIRIDIS colormap to a 2D feature map."""
    # Normalize to 0-255
    min_val = feature_map_2d.min()
    max_val = feature_map_2d.max()
    
    if max_val == min_val:
        normalized = np.zeros_like(feature_map_2d, dtype=np.uint8)
    else:
        normalized = (255 * (feature_map_2d - min_val) / (max_val - min_val)).astype(np.uint8)
        
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    return heatmap

def resize_image(arr: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Resize while maintaining aspect ratio."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_size:
        return arr
    
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
        
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_detections(image: np.ndarray, detections: list, class_names: dict) -> np.ndarray:
    """Draw bounding boxes with labels on an image."""
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
        class_id = det.get('class_id', 0)
        conf = det.get('confidence', 0.0)
        
        # Draw bounding box
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Label text
        class_name = class_names.get(class_id, str(class_id))
        label = f"{class_name} {conf:.2f}"
        
        # Text background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return img_copy
