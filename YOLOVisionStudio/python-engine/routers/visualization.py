import os
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from models.schemas import ArchitectureRequest, ArchitectureResponse, LayerInfo, FeatureMapsRequest, FeatureMapsResponse, FeatureMap
from ultralytics import YOLO
from utils.image_utils import apply_heatmap, numpy_to_base64, base64_to_numpy, resize_image
from utils.model_resolver import resolve_model_path

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
