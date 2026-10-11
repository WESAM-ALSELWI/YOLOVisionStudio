import os
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import APIRouter, HTTPException
from models.schemas import (
    ArchitectureRequest, ArchitectureResponse, LayerInfo,
    FeatureMapsRequest, FeatureMapsResponse, FeatureMap,
    LayerStatistics, WeightVisualization,
    GradCAMRequest, GradCAMResponse,
    AttentionMapRequest, AttentionMapResponse
)
from ultralytics import YOLO
from utils.image_utils import apply_heatmap, numpy_to_base64, base64_to_numpy, resize_image
from utils.model_resolver import resolve_model_path
import cv2
import base64

router = APIRouter(prefix="/visualization", tags=["Visualization"])

# ---------------------------------------------------------------------------
# Beginner-friendly, plain-English description for each module type.
# ---------------------------------------------------------------------------
PLAIN_ENGLISH = {
    "Conv": "Feature detector — slides small filters over the image to find edges, colors and patterns",
    "Conv2d": "The core math: multiplies filters over the image to extract features",
    "BatchNorm2d": "Normalizer — keeps the numbers in each layer stable so training doesn't explode",
    "SiLU": "Activation — keeps positive signals, softly drops negative ones (lets the net learn non-linear things)",
    "ReLU": "Activation — passes positives only",
    "LeakyReLU": "Activation — passes positives, shrinks negatives",
    "Sigmoid": "Squashes any number into 0..1 (used for confidence scores)",
    "MaxPool2d": "Max pooling — shrinks the map by keeping only the strongest activation in each window",
    "AvgPool2d": "Average pooling — shrinks the map by averaging each window",
    "AdaptiveAvgPool2d": "Averages everything down to one value per channel",
    "Flatten": "Unrolls the feature grid into a single long vector",
    "Linear": "Fully connected layer — every input connects to every output (classic neural network math)",
    "C2f": "Feature combiner — splits the data, refines it through small conv stacks, then merges all paths (YOLO's building block)",
    "C3": "Older feature combiner — like C2f but with 3 paths",
    "C3k2": "Feature combiner — C2f variant using smaller kernels (v11 backbone block)",
    "C3k": "Small-kernel feature combiner used inside C3k2",
    "Bottleneck": "Mini-block: shrink channels, process, expand back — learns deeper patterns cheaply",
    "SPPF": "Multi-scale look — applies max pooling 3 times at different sizes so the net sees big AND small objects",
    "Upsample": "Zooms the feature map back up (e.g. 20x20 → 40x40) so fine details are recovered",
    "Concat": "Stitches two feature maps together channel-wise (combines deep + shallow info)",
    "Detect": "The head — predicts final boxes: where objects are + class + confidence",
    "Segment": "Segmentation head — predicts boxes plus pixel masks",
    "Pose": "Pose head — predicts keypoints on detected objects",
    "OBB": "Rotated-box head — predicts rotated bounding boxes",
    "Classify": "Classification head — picks ONE label for the whole image (this is where fully-connected-style decisions happen)",
    "DFL": "Distribution focal loss layer — turns box-edge predictions into precise coordinates",
    "Identity": "Pass-through — nothing changes, just wiring",
}

# Layers that contain a real Linear (fully connected) op worth calling out.
_LINEAR_TYPES = {"Linear"}


def _shape_list(t) -> list:
    try:
        return [int(x) for x in t.shape]
    except Exception:
        return []


def _module_extras(module):
    """(kernel, stride, details, flops_hint_fn) extracted from common modules."""
    kernel = stride = details = None

    def _get(obj, attr):
        v = getattr(obj, attr, None)
        return tuple(v) if isinstance(v, (list, tuple)) else v

    if module.__class__.__name__ in ("Conv2d", "MaxPool2d", "AvgPool2d"):
        k = _get(module, "kernel_size")
        s = _get(module, "stride")
        p = _get(module, "padding")
        if k is not None:
            kernel = str(k)
        if s is not None:
            stride = str(s)
        bits = []
        if hasattr(module, "in_channels"):
            bits.append(f"in={getattr(module, 'in_channels')}, out={getattr(module, 'out_channels')}")
        if k is not None:
            bits.append(f"kernel={k}")
        if s is not None:
            bits.append(f"stride={s}")
        if p is not None:
            bits.append(f"pad={p}")
        details = ", ".join(bits)
    elif module.__class__.__name__ == "Linear":
        details = f"in_features={getattr(module, 'in_features', '?')}, out_features={getattr(module, 'out_features', '?')}"
    else:
        # Ultralytics wrappers (Conv, C2f, SPPF, Detect, ...) expose cv1/cv2/conv
        convs = [m for m in module.children() if isinstance(m, torch.nn.Conv2d)]
        if convs:
            c = convs[0]
            kernel, stride = str(tuple(c.kernel_size)), str(tuple(c.stride))
            details = f"in={c.in_channels}, out={c.out_channels}, kernel={tuple(c.kernel_size)}, stride={tuple(c.stride)}"

    return kernel, stride, details


def _conv_flops_g(module, out_shape) -> float:
    """Rough Conv2d FLOPs estimate in GFLOPs from the captured output shape."""
    try:
        c = module if isinstance(module, torch.nn.Conv2d) else getattr(module, "conv", None)
        if c is None or not isinstance(c, torch.nn.Conv2d) or len(out_shape) != 4:
            return 0.0
        _, out_ch, oh, ow = out_shape
        elems = out_ch * oh * ow
        flops = 2.0 * elems * (c.in_channels / c.groups) * c.kernel_size[0] * c.kernel_size[1]
        return round(flops / 1e9, 4)
    except Exception:
        return 0.0


@router.get("/architecture/{yolo_version}/{model_size}", response_model=ArchitectureResponse)
async def get_architecture(yolo_version: str, model_size: str):
    """Return the FULL recursive architecture of a YOLO model.

    Every nested sub-layer is included (Conv -> Conv2d+BatchNorm+SiLU,
    SPPF -> MaxPool2d x3, Detect -> per-scale branches). Real input/output
    tensor shapes are captured by running one dummy forward pass with hooks,
    so users see exactly how the image transforms step-by-step end-to-end."""
    model_name = f"yolo{yolo_version.replace('v', '')}{model_size}.pt"
    try:
        model_path = resolve_model_path(model_name)
        model = YOLO(model_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}: {str(e)}")

    pytorch_model = model.model

    # ---- capture REAL shapes with forward hooks on EVERY module ----
    shapes_in, shapes_out = {}, {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if hasattr(t, "shape"):
                shapes_out[name] = _shape_list(t)
            if inp and hasattr(inp[0], "shape"):
                shapes_in[name] = _shape_list(inp[0])
        return hook

    for name, mod in pytorch_model.named_modules():
        if name:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    try:
        import torch as _t
        pytorch_model.eval()
        with _t.no_grad():
            pytorch_model(_t.zeros(1, 3, 640, 640))
    except Exception:
        pass  # shapes stay empty but the tree is still returned
    finally:
        for h in hooks:
            h.remove()

    # ---- build the recursive tree from named_modules ----
    nodes: dict = {}
    root_children = []
    total_params = sum(p.numel() for p in pytorch_model.parameters())

    named = [(n, m) for n, m in pytorch_model.named_modules() if n]

    def build(name, module) -> LayerInfo:
        mtype = module.__class__.__name__
        params = sum(p.numel() for p in module.parameters())
        kernel, stride, details = _module_extras(module)
        out_shape = shapes_out.get(name)
        flops = _conv_flops_g(module, out_shape) if out_shape else 0.0
        act = None
        if mtype == "Conv":
            inner = [c.__class__.__name__ for c in module.children()]
            act = next((a for a in inner if a in ("SiLU", "ReLU", "LeakyReLU")), None)
        node = LayerInfo(
            id=name,
            name=name.split("/")[-1],
            type=mtype,
            input_shape=shapes_in.get(name),
            output_shape=out_shape,
            parameters=params,
            flops=flops,
            activation=act,
            kernel=kernel,
            stride=stride,
            details=details,
            plain=PLAIN_ENGLISH.get(mtype),
        )
        # recurse into children (skip containers of raw params like Conv2d itself)
        kids = [(cn, cm) for cn, cm in named if cn.startswith(name + ".") and "." not in cn[len(name) + 1:]]
        node.children = [build(cn, cm) for cn, cm in kids]
        return node

    top = [(n, m) for n, m in named if "." not in n]
    layers = [build(n, m) for n, m in top]

    total_flops = 0.0
    def walk(ls):
        nonlocal total_flops
        for l in ls:
            total_flops += l.flops
            walk(l.children)
    walk(layers)

    return ArchitectureResponse(
        layers=layers,
        total_parameters=total_params,
        total_flops=round(total_flops, 2),
    )

@router.post("/feature-maps", response_model=FeatureMapsResponse)
async def get_feature_maps(request: FeatureMapsRequest):
    """Extract feature maps from intermediate layers."""
    model_path = resolve_model_path(request.model_path)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    try:
        model = YOLO(model_path)
        
        # Load image
        if request.image_base64:
            img = base64_to_numpy(request.image_base64)
        elif request.image_path and os.path.exists(request.image_path):
            import cv2
            img = cv2.imread(request.image_path)
        else:
            raise HTTPException(status_code=400, detail="Provide image_base64 or valid image_path")
            
        img = resize_image(img, max_size=640)
        
        # Setup hooks
        activations = {}
        hooks = []
        
        def make_hook(layer_idx, name):
            def hook(module, input, output):
                # Detach and move to CPU
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                activations[layer_idx] = out.detach().cpu().numpy()
            return hook
            
        # Register hooks
        target_layers = request.layer_indices
        pytorch_model = model.model
        
        module_list = list(pytorch_model.children())
        for i, module in enumerate(module_list):
            if i in target_layers:
                h = module.register_forward_hook(make_hook(i, module.__class__.__name__))
                hooks.append(h)
                
        # Run forward pass (model.predict automatically prepares input)
        model.predict(source=img, verbose=False)
        
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Process activations
        result_maps = []
        for idx in target_layers:
            if idx in activations:
                act = activations[idx]
                if act.ndim == 4:
                    # Batch, Channels, Height, Width
                    channels = act[0]
                    # Take up to 16 channels to prevent massive payloads
                    max_channels = min(16, channels.shape[0])
                    
                    channel_b64s = []
                    for c in range(max_channels):
                        feat = channels[c]
                        heatmap = apply_heatmap(feat)
                        channel_b64s.append(numpy_to_base64(heatmap))
                        
                    result_maps.append(FeatureMap(
                        layer_index=idx,
                        layer_name=module_list[idx].__class__.__name__,
                        shape=list(act.shape),
                        channels_base64=channel_b64s
                    ))
                    
        return FeatureMapsResponse(
            original_image_base64=numpy_to_base64(img),
            feature_maps=result_maps
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Real XAI: Grad-CAM (Gradient-weighted Class Activation Mapping)
# ---------------------------------------------------------------------------
@router.post("/gradcam", response_model=GradCAMResponse)
async def generate_gradcam(request: GradCAMRequest):
    """Generate Grad-CAM heatmap to show what the model focuses on."""
    model_path = resolve_model_path(request.model_path)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = YOLO(model_path)
        pytorch_model = model.model

        # Load image
        if request.image_base64:
            img = base64_to_numpy(request.image_base64)
        elif request.image_path and os.path.exists(request.image_path):
            img = cv2.imread(request.image_path)
        else:
            raise HTTPException(status_code=400, detail="Provide image_base64 or valid image_path")

        original_img = img.copy()
        img_resized = resize_image(img, max_size=640)

        # Prepare input tensor
        input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(next(pytorch_model.parameters()).device)

        # Find target layer
        target_layer = None
        for name, module in pytorch_model.named_modules():
            if request.target_layer in name:
                target_layer = module
                break

        if target_layer is None:
            # Try to find by index
            layers = list(pytorch_model.modules())
            try:
                idx = int(request.target_layer)
                if idx < len(layers):
                    target_layer = layers[idx]
            except:
                pass

        if target_layer is None:
            raise HTTPException(status_code=400, detail=f"Target layer '{request.target_layer}' not found")

        # Register hooks for gradients and activations
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0].detach()

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        pytorch_model.eval()
        output = pytorch_model(input_tensor)

        # Get target class prediction
        if request.class_id is not None:
            target_class = request.class_id
        else:
            # Use highest confidence class
            if hasattr(output, '__iter__') and len(output) > 0:
                out = output[0] if isinstance(output, (list, tuple)) else output
                if hasattr(out, 'max'):
                    target_class = out.max(dim=1)[1].item()
                else:
                    target_class = 0
            else:
                target_class = 0

        # Backward pass to get gradients
        pytorch_model.zero_grad()
        if hasattr(output, '__iter__') and len(output) > 0:
            out = output[0] if isinstance(output, (list, tuple)) else output
            if out.ndimension() > 1:
                out[0, target_class].backward(retain_graph=True)
            else:
                out.backward(retain_graph=True)

        # Remove hooks
        handle_fwd.remove()
        handle_bwd.remove()

        # Generate Grad-CAM
        if 'value' not in activations or 'value' not in gradients:
            raise HTTPException(status_code=500, detail="Failed to capture activations/gradients")

        act = activations['value'].cpu()
        grad = gradients['value'].cpu()

        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=(img_resized.shape[0], img_resized.shape[1]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # Create overlay
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', original_img)
        original_b64 = base64.b64encode(buffer).decode('ascii')

        _, buffer = cv2.imencode('.jpg', heatmap)
        heatmap_b64 = base64.b64encode(buffer).decode('ascii')

        _, buffer = cv2.imencode('.jpg', overlay)
        overlay_b64 = base64.b64encode(buffer).decode('ascii')

        return GradCAMResponse(
            original_image_base64=original_b64,
            heatmap_base64=heatmap_b64,
            overlay_base64=overlay_b64,
            target_class=str(target_class),
            confidence=float(output[0, target_class].item()) if hasattr(output, '__iter__') else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Real XAI: Attention Maps (Multi-scale feature visualization)
# ---------------------------------------------------------------------------
@router.post("/attention-maps", response_model=AttentionMapResponse)
async def generate_attention_maps(request: AttentionMapRequest):
    """Generate multi-scale attention maps showing what features the network focuses on."""
    model_path = resolve_model_path(request.model_path)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = YOLO(model_path)

        # Load image
        if request.image_base64:
            img = base64_to_numpy(request.image_base64)
        elif request.image_path and os.path.exists(request.image_path):
            img = cv2.imread(request.image_path)
        else:
            raise HTTPException(status_code=400, detail="Provide image_base64 or valid image_path")

        original_img = img.copy()
        img_resized = resize_image(img, max_size=640)

        # Capture multi-scale feature maps
        attention_maps = []
        hooks = []
        feature_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, (list, tuple)):
                    out = output[0]
                else:
                    out = output
                if hasattr(out, 'shape') and out.ndimension() == 4:
                    feature_outputs[layer_idx] = out.detach()
            return hook

        # Register hooks on backbone layers
        pytorch_model = model.model
        for idx, module in enumerate(pytorch_model.modules()):
            if isinstance(module, torch.nn.Conv2d) and idx % 3 == 0:  # Sample every 3rd conv
                h = module.register_forward_hook(make_hook(idx))
                hooks.append(h)

        # Forward pass
        model.predict(source=img_resized, verbose=False)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Process feature outputs
        for idx, feat in feature_outputs.items():
            if feat.ndimension() == 4:
                # Global average pooling to get attention weights
                attention = feat.mean(dim=1, keepdim=True)  # [1, 1, H, W]
                attention = F.relu(attention)

                # Resize to original size
                attention_resized = F.interpolate(
                    attention,
                    size=(original_img.shape[0], original_img.shape[1]),
                    mode='bilinear',
                    align_corners=False
                )
                attention_resized = attention_resized.squeeze().cpu().numpy()

                # Normalize
                attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)

                # Create heatmap
                heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                _, buffer = cv2.imencode('.jpg', heatmap)
                heatmap_b64 = base64.b64encode(buffer).decode('ascii')

                attention_maps.append({
                    "layer_index": idx,
                    "shape": list(feat.shape),
                    "heatmap_base64": heatmap_b64
                })

        # Create combined overlay
        if attention_maps:
            # Use the last (most semantic) attention map
            last_attention = cv2.imdecode(
                np.frombuffer(base64.b64decode(attention_maps[-1]["heatmap_base64"]), np.uint8),
                cv2.IMREAD_COLOR
            )
            overlay = cv2.addWeighted(original_img, 0.5, last_attention, 0.5, 0)
            _, buffer = cv2.imencode('.jpg', overlay)
            overlay_b64 = base64.b64encode(buffer).decode('ascii')
        else:
            overlay_b64 = ""

        _, buffer = cv2.imencode('.jpg', original_img)
        original_b64 = base64.b64encode(buffer).decode('ascii')

        return AttentionMapResponse(
            original_image_base64=original_b64,
            attention_maps=attention_maps,
            overlay_base64=overlay_b64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attention map generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Real Weight/Filter Visualization
# ---------------------------------------------------------------------------
@router.get("/weights/{model_path:path}")
async def visualize_weights(model_path: str, layer_index: int = 0):
    """Visualize learned weights/filters of a specific layer."""
    resolved_path = resolve_model_path(model_path)
    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = YOLO(resolved_path)
        pytorch_model = model.model

        # Get the target layer
        layers = list(pytorch_model.modules())
        conv_layers = [m for m in layers if isinstance(m, torch.nn.Conv2d)]

        if layer_index >= len(conv_layers):
            raise HTTPException(status_code=400, detail=f"Layer index {layer_index} out of range")

        target_layer = conv_layers[layer_index]

        # Get weights
        weights = target_layer.weight.data.cpu().numpy()
        weight_shape = list(weights.shape)

        # Calculate statistics
        weight_stats = {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max())
        }

        # Create grid visualization of filters
        # For conv layers: [out_channels, in_channels, kernel_h, kernel_w]
        out_ch, in_ch, kh, kw = weight_shape

        # Take first 16 output channels and first 3 input channels (RGB-like)
        display_ch = min(16, out_ch)
        display_in = min(3, in_ch)

        # Normalize each filter for visualization
        grid_rows = int(np.ceil(display_ch / 4))
        grid_cols = min(4, display_ch)

        grid = np.zeros((grid_rows * kh, grid_cols * kw), dtype=np.float32)

        for i in range(display_ch):
            row = i // 4
            col = i % 4

            # Average across input channels for single-channel visualization
            filter_img = weights[i, :display_in].mean(axis=0)

            # Normalize
            f_min, f_max = filter_img.min(), filter_img.max()
            if f_max - f_min > 1e-8:
                filter_img = (filter_img - f_min) / (f_max - f_min)

            grid[row * kh:(row + 1) * kh, col * kw:(col + 1) * kw] = filter_img

        # Convert to uint8 image
        grid = (grid * 255).astype(np.uint8)
        grid = cv2.resize(grid, (grid_cols * 64, grid_rows * 64), interpolation=cv2.INTER_NEAREST)

        # Apply colormap
        grid_colored = cv2.applyColorMap(grid, cv2.COLORMAP_INFERNO)
        _, buffer = cv2.imencode('.jpg', grid_colored)
        weights_b64 = base64.b64encode(buffer).decode('ascii')

        # Bias visualization if exists
        bias_b64 = None
        if target_layer.bias is not None:
            bias = target_layer.bias.data.cpu().numpy()
            bias_hist = np.histogram(bias, bins=50)[0]
            # Simple bar chart visualization
            max_val = bias_hist.max()
            if max_val > 0:
                bias_hist = (bias_hist / max_val * 255).astype(np.uint8)
                bias_img = np.zeros((100, len(bias_hist)), dtype=np.uint8)
                for i, h in enumerate(bias_hist):
                    bias_img[:h, i] = 255
                bias_img = cv2.applyColorMap(bias_img, cv2.COLORMAP_BLUES)
                _, buffer = cv2.imencode('.jpg', bias_img)
                bias_b64 = base64.b64encode(buffer).decode('ascii')

        return WeightVisualization(
            layer_index=layer_index,
            layer_name=f"Conv2d_Layer_{layer_index}",
            weight_shape=weight_shape,
            weights_base64=weights_b64,
            weight_stats=weight_stats,
            bias_base64=bias_b64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weight visualization failed: {str(e)}")


# ---------------------------------------------------------------------------
# Real-time Layer Statistics During Training
# ---------------------------------------------------------------------------
@router.get("/layer-statistics/{job_id}")
async def get_layer_statistics(job_id: str):
    """Get real-time statistics for all layers during training."""
    try:
        from utils.training_viz import TrainingVisualizer, compute_all_layer_statistics
        
        # Get the training visualizer for this job
        # This would typically be stored in a global registry
        # For now, we'll return a placeholder
        return {
            "job_id": job_id,
            "layer_statistics": [],
            "message": "Layer statistics endpoint - requires active training job"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get layer statistics: {str(e)}")


@router.get("/training-snapshot/{job_id}")
async def get_training_snapshot(job_id: str):
    """Get a comprehensive snapshot of training state including layer statistics."""
    try:
        from utils.training_viz import TrainingVisualizer, compute_all_layer_statistics
        
        # This would retrieve the active training visualizer
        # For now, return structure
        return {
            "job_id": job_id,
            "epoch": 0,
            "batch": 0,
            "layer_statistics": [],
            "grad_norms": {},
            "activation_heatmaps": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training snapshot: {str(e)}")
