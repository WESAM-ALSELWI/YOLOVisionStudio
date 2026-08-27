"""Real-time instrumentation of an actual Ultralytics training run.

Everything captured here comes from the live trainer object: real loss
components, learning rate, the real augmented batch images, ground-truth
labels, genuine network predictions and true per-layer activations for the
WHOLE network (CNN-Explainer style). Heavy captures are throttled so
training speed is barely affected.
"""
import os
import time
import threading

import numpy as np

# How often (seconds) expensive captures run. Configurable without code changes.
try:
    CAPTURE_INTERVAL = float(os.environ.get("YOLO_VIZ_CAPTURE_INTERVAL", "2.5"))
except ValueError:
    CAPTURE_INTERVAL = 2.5

FRAME_MAX_WIDTH = 512
PIPELINE_CHANNELS = 8      # feature maps per layer column (CNN-Explainer style)
PIPELINE_SIZE = 56         # thumbnail size (px) in the flow view
DETAIL_CHANNELS = 8        # bigger maps when user clicks a layer
DETAIL_SIZE = 96

# Diverging red->white->blue LUT (like CNN Explainer): negative = red,
# zero = near white, positive = blue.
_DIV_LUT = None
def _diverging_lut():
    global _DIV_LUT
    if _DIV_LUT is None:
        import cv2
        neg = np.array([203, 59, 59], dtype=np.float32)    # BGR red-ish
        mid = np.array([248, 248, 251], dtype=np.float32)  # BGR near-white
        pos = np.array([246, 135, 59], dtype=np.float32)   # BGR blue-ish
        xs = np.linspace(0, 1, 128)
        lower = np.stack([neg[0] + (mid[0] - neg[0]) * xs,
                          neg[1] + (mid[1] - neg[1]) * xs,
                          neg[2] + (mid[2] - neg[2]) * xs], axis=1)
        upper = np.stack([mid[0] + (pos[0] - mid[0]) * xs,
                          mid[1] + (pos[1] - mid[1]) * xs,
                          mid[2] + (pos[2] - mid[2]) * xs], axis=1)
        _DIV_LUT = np.concatenate([lower, upper], axis=0).astype(np.uint8)  # 256x3 BGR
    return _DIV_LUT


def _thumb(fmap: np.ndarray, size: int, quality: int = 60) -> str | None:
    """Convert a single 2D activation map into a base64 JPEG heatmap."""
    import cv2, base64
    try:
        f = np.asarray(fmap, dtype=np.float32)
        if f.size == 0 or not np.isfinite(f).any():
            return None
        mn, mx = float(f.min()), float(f.max())
        if mx - mn > 1e-9:
            f = (f - mn) / (mx - mn)
        else:
            f = np.zeros_like(f)
        small = cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap((small * 255).astype(np.uint8), cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def _thumb_div(fmap: np.ndarray, size: int, quality: int = 62):
    """Diverging red-white-blue render. Returns (b64, vmin, vmax) using the
    ACTUAL min/max of the map so the UI can print a real color scale."""
    import cv2, base64
    try:
        f = np.asarray(fmap, dtype=np.float32)
        if f.size == 0 or not np.isfinite(f).any():
            return None, 0.0, 0.0
        vmax = float(np.abs(f).max())
        if vmax < 1e-9:
            f = np.full_like(f, 0.5)
        else:
            f = f / vmax          # -> [-1, 1]
        small = cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST)
        idx = ((small + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
        img = _diverging_lut()[idx]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, 0.0, 0.0
        mn = float(np.asarray(fmap).min())
        mx = float(np.asarray(fmap).max())
        return base64.b64encode(buf.tobytes()).decode("ascii"), round(mn, 3), round(mx, 3)
    except Exception:
        return None, 0.0, 0.0


def _tensor_maps(act):
    """Yield 2D numpy maps from an arbitrary module output."""
    import torch
    import torch.nn.functional as F

    def prep(t):
        # t: CHW tensor -> list of channel maps, pooled to square
        t = t.detach().float().cpu()
        t = F.adaptive_avg_pool2d(t.unsqueeze(0), (PIPELINE_SIZE * 2, PIPELINE_SIZE * 2))[0]
        return t.numpy()

    if isinstance(act, dict):
        # Some heads return dicts of tensors (e.g. one2one / multi-scale preds)
        tensors = [v for v in act.values() if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 3]
        scales = tensors[:3]
        out = []
        for s in scales:
            m = s[0] if s.ndim == 4 else s
            out.append(m.mean(dim=0).detach().float().cpu().numpy())
        return out, True
    if isinstance(act, (list, tuple)):
        # Detection head returns multiple scale maps [B,C,H,W] each
        scales = [a for a in act if hasattr(a, "shape") and a.ndim >= 3][:3]
        out = []
        for s in scales:
            m = s[0] if s.ndim == 4 else s
            out.append(m.mean(dim=0).detach().float().cpu().numpy())
        return out, True
    if not hasattr(act, "shape"):
        return [], False
    t = act[0] if act.ndim == 4 else act          # drop batch dim
    if t.ndim < 3:
        return [t.detach().float().cpu().numpy()], False
    return [c for c in prep(t)], False


class TrainingVisualizer:
    """Attached to one YOLO training job via ultralytics callbacks."""

    def __init__(self, feature_layers=None):
        self.lock = threading.Lock()
        self.status = "running"
        self.error_message = ""

        # Per-batch real metrics
        self.metrics = {
            "epoch": 0, "batch": 0, "num_batches": 0,
            "box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0,
            "lr": 0.0,
            "map50": None, "map5095": None, "precision": None, "recall": None,
        }

        # Latest captured artifacts (seq numbers let WS clients detect updates)
        self.frame_seq = 0
        self.pipeline_seq = 0
        self.grad_seq = 0
        self.frame = None       # {image_base64, gt:[...], pred:[...], ...}
        self.pipeline = None    # ordered list of stage dicts for every layer
        self.grad_norms = {}    # layer idx -> L2 norm of weight gradient (backprop!)
        self.class_names = {}
        self.layer_meta = {}    # idx -> {name, params, from, activation, conv, bn, inner...}

        self._last_capture = 0.0
        self._hooks = []
        self._activations = {}       # latest raw forward output per layer
        self._sub_activations = {}   # inner conv/bn outputs of layer 0 (decomposition)
        self._grad_activations = {}  # latest real BACKWARD gradient per layer
        self._modules_by_idx = {}    # live module references (for weight access)

        # Real batch data captured during the live loop
        self._last_batch = None
        self._batch_counter = 0
        self.save_dir = ""

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------
    def attach(self, model):
        """Register ultralytics callbacks on the live model wrapper."""
        try:
            names = getattr(model, "names", None)
            if isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

        model.add_callback("on_train_batch_end", self._on_train_batch_end)
        model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        model.add_callback("on_train_start", self._on_train_start)
        model.add_callback("on_train_end", self._on_train_end)

    def detach(self, model):
        try:
            for ev in ("on_train_batch_end", "on_fit_epoch_end", "on_train_start", "on_train_end"):
                model.clear_callback(ev)
        except Exception:
            pass
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Ultralytics callbacks (receive the live trainer only)
    # ------------------------------------------------------------------
    def _on_train_start(self, trainer):
        """Called once when the real training loop starts. Model.train()
        rebuilds the nn.Module before this point, so hooks go here. The batch
        dict is only visible inside the loop, so we wrap preprocess_batch."""
        try:
            self.save_dir = str(getattr(trainer, "save_dir", ""))
            try:
                self.metrics["num_batches"] = int(len(trainer.train_loader))
            except Exception:
                pass
        except Exception:
            pass

        # Register activation hooks on EVERY layer of the REAL training net.
        try:
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._hooks.clear()
            net = trainer.model
            layers = net.model if hasattr(net, "model") else net  # DetectionModel -> nn.Sequential
            self.layer_meta = {}
            for idx, module in enumerate(layers):
                meta = self._module_info(idx, module)

                # BatchNorm info (Conv layers in YOLO = Conv + BN + activation)
                bn = getattr(module, "bn", None)
                if bn is not None and hasattr(bn, "__class__"):
                    meta["bn"] = {"type": bn.__class__.__name__, "affine": bool(getattr(bn, "affine", False))}

                # Inner building blocks (pooling / dropout / linear inside blocks)
                inner = {"MaxPool2d": 0, "AvgPool2d": 0, "Dropout": 0, "Dropout2d": 0, "Linear": 0}
                for _, sm in module.named_modules():
                    n = sm.__class__.__name__
                    if n in inner:
                        inner[n] += 1
                meta["inner"] = {k: v for k, v in inner.items() if v}

                self.layer_meta[idx] = meta
                self._modules_by_idx[idx] = module
                handle = module.register_forward_hook(self._make_hook(idx))
                self._hooks.append(handle)

            # Decompose layer 0 into its REAL inner steps: raw conv -> BN.
            m0 = self._modules_by_idx.get(0)
            if m0 is not None:
                inner_conv = getattr(m0, "conv", None)
                inner_bn = getattr(m0, "bn", None)
                if inner_conv is not None:
                    self._hooks.append(inner_conv.register_forward_hook(self._make_sub_hook("0c")))
                if inner_bn is not None:
                    self._hooks.append(inner_bn.register_forward_hook(self._make_sub_hook("0b")))
        except Exception as e:
            print(f"[viz] hook registration error: {e}", flush=True)

        # Capture class names from the live model
        try:
            names = getattr(trainer.model, "names", None)
            if isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

        # Wrap preprocess_batch so every REAL augmented batch is captured.
        try:
            orig_preprocess = trainer.preprocess_batch

            def wrapped_preprocess(batch):
                out = orig_preprocess(batch)
                try:
                    self._last_batch = out if isinstance(out, dict) else batch
                    self._batch_counter += 1
                except Exception:
                    pass
                return out

            trainer.preprocess_batch = wrapped_preprocess
        except Exception as e:
            print(f"[viz] could not wrap preprocess_batch: {e}", flush=True)

    def _module_info(self, idx: int, module) -> dict:
        """Extract REAL architecture facts from the live module: its actual
        input connections (from), activation function, weight/bias shapes."""
        import torch.nn as nn

        params = 0
        try:
            params = sum(p.numel() for p in module.parameters())
        except Exception:
            pass

        info = {
            "name": module.__class__.__name__,
            "params": int(params),
            "is_head": "Detect" in module.__class__.__name__ or "Segment" in module.__class__.__name__,
            # Real input connections (Ultralytics stores source indices in .f)
            "from": [],
        }
        frm = getattr(module, "f", -1)
        if isinstance(frm, (list, tuple)):
            info["from"] = [int(x) for x in frm]
        elif isinstance(frm, int) and frm >= 0:
            info["from"] = [frm]

        # Activation function actually used by this layer
        act = getattr(module, "act", None)
        if act is not None and hasattr(act, "__class__"):
            info["activation"] = act.__class__.__name__

        # Weight / bias facts for convolutional layers
        conv = getattr(module, "conv", None)
        target = conv if isinstance(conv, nn.modules.conv._ConvNd) else (
            module if isinstance(module, nn.modules.conv._ConvNd) else None)
        if target is not None:
            w = getattr(target, "weight", None)
            b = getattr(target, "bias", None)
            info["conv"] = {
                "in_channels": int(getattr(target, "in_channels", 0)),
                "out_channels": int(getattr(target, "out_channels", 0)),
                "kernel": list(target.kernel_size) if hasattr(target, "kernel_size") else None,
                "stride": list(target.stride) if hasattr(target, "stride") else None,
                "padding": list(target.padding) if hasattr(target, "padding") else None,
                "weight_shape": [int(x) for x in w.shape] if w is not None else None,
                "weight_count": int(w.numel()) if w is not None else 0,
                "has_bias": b is not None,
                "bias_count": int(b.numel()) if b is not None else 0,
                # Live weight statistics (mean/std of the actual tensors)
                "weight_mean": round(float(w.detach().float().mean()), 5) if w is not None else None,
                "weight_std": round(float(w.detach().float().std()), 5) if w is not None else None,
            }
        return info

    def _make_hook(self, idx):
        """Forward hook: store the real output tensor AND attach a gradient
        hook to it so we capture the BACKWARD pass through this layer too."""
        def hook(_m, _inp, out):
            try:
                t = out
                if isinstance(t, dict):
                    t = next(iter(t.values()))
                elif isinstance(t, (list, tuple)):
                    t = t[0] if len(t) > 0 else None
                if t is None or not hasattr(t, "shape") or getattr(t, "ndim", 0) < 3:
                    return
                self._activations[idx] = t
                try:
                    if getattr(t, "requires_grad", False):
                        t.register_hook(lambda g, i=idx: self._store_grad(i, g))
                except Exception:
                    pass
            except Exception:
                pass
        return hook

    def _make_sub_hook(self, key: str):
        """Hook on an inner module (raw conv / batchnorm of layer 0)."""
        def hook(_m, _inp, out):
            try:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if hasattr(t, "shape") and getattr(t, "ndim", 0) >= 3:
                    self._sub_activations[key] = t
            except Exception:
                pass
        return hook

    def _store_grad(self, idx, g):
        """Called by autograd DURING backpropagation with the real dL/dout."""
        try:
            self._grad_activations[idx] = g.detach()
        except Exception:
            pass

    def _on_train_batch_end(self, trainer):
        try:
            m = {}
            m["epoch"] = int(getattr(trainer, "epoch", 0)) + 1
            m["batch"] = self._batch_counter
            m["num_batches"] = self.metrics.get("num_batches", 0)

            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                vals = tloss.detach().cpu().flatten().tolist()
                if len(vals) >= 3:
                    m["box_loss"], m["cls_loss"], m["dfl_loss"] = (
                        round(float(vals[0]), 4), round(float(vals[1]), 4), round(float(vals[2]), 4))
                elif len(vals) > 0:
                    m["box_loss"] = round(float(vals[0]), 4)

            opt = getattr(trainer, "optimizer", None)
            if opt is not None and len(opt.param_groups) > 0:
                m["lr"] = float(opt.param_groups[0]["lr"])

            with self.lock:
                self.metrics.update(m)

            # ---- REAL BACKPROP VISIBILITY: L2 norm of each layer's weight
            # gradients right after loss.backward() (they are populated now).
            try:
                import math
                gn = {}
                for i, mod in self._modules_by_idx.items():
                    s, n = 0.0, 0
                    for p in mod.parameters():
                        g = getattr(p, "grad", None)
                        if g is not None:
                            s += float(g.detach().float().norm()) ** 2
                            n += 1
                    if n:
                        gn[i] = round(math.sqrt(s), 5)
                if gn:
                    with self.lock:
                        self.grad_norms = gn
                        self.grad_seq += 1
            except Exception:
                pass

            now = time.time()
            if now - self._last_capture >= CAPTURE_INTERVAL:
                self._last_capture = now
                self._capture_frame(trainer)
                self._package_pipeline()
        except Exception as e:
            # Instrumentation must never break training.
            print(f"[viz] train_batch_end instrumentation error: {e}", flush=True)

    def _on_fit_epoch_end(self, trainer):
        try:
            metrics = getattr(trainer, "metrics", None) or {}
            upd = {}
            for key, target in (
                ("metrics/mAP50(B)", "map50"),
                ("metrics/mAP50-95(B)", "map5095"),
                ("metrics/precision(B)", "precision"),
                ("metrics/recall(B)", "recall"),
            ):
                if key in metrics:
                    upd[target] = round(float(metrics[key]), 4)
            with self.lock:
                self.metrics.update(upd)
        except Exception:
            pass

    def _on_train_end(self, trainer):
        with self.lock:
            self.status = "completed"

    # ------------------------------------------------------------------
    # Real captures
    # ------------------------------------------------------------------
    def _capture_frame(self, trainer):
        import cv2
        import torch

        batch = self._last_batch
        if not batch or "img" not in batch:
            return

        imgs = batch["img"]
        if not hasattr(imgs, "__getitem__"):
            return
        img_t = imgs[0].detach().float().cpu()          # CHW, 0..1
        _, h, w = img_t.shape
        arr = (img_t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr[:, :, ::-1])      # RGB -> BGR for OpenCV

        scale = FRAME_MAX_WIDTH / max(arr.shape[:2])
        if scale < 1:
            arr = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)))
        fx, fy = arr.shape[1] / w, arr.shape[0] / h

        # Ground truth (real labels for this exact augmented image)
        gt = []
        try:
            def to_np(x):
                if hasattr(x, "detach"):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            bidx = batch.get("batch_idx")
            cls_t = batch.get("cls")
            bboxes = batch.get("bboxes")
            if bidx is not None and cls_t is not None and bboxes is not None:
                bidx_np = to_np(bidx).flatten()
                cls_np = to_np(cls_t).flatten()
                box_np = to_np(bboxes).reshape(-1, 4)
                mask = bidx_np == 0
                names = self.class_names
                for ci, (xc, yc, bw, bh) in zip(cls_np[mask], box_np[mask]):
                    gt.append({
                        "class_id": int(ci),
                        "name": names.get(int(ci), str(int(ci))),
                        # normalized cx,cy,w,h -> pixels on displayed frame
                        "bbox": [
                            round((xc - bw / 2) * w, 1),
                            round((yc - bh / 2) * h, 1),
                            round(bw * w, 1),
                            round(bh * h, 1),
                        ],
                    })
        except Exception as e:
            print(f"[viz] gt extraction error: {e}", flush=True)

        # Genuine predictions from the current weights (throttled, no_grad)
        pred = []
        try:
            try:
                from ultralytics.utils.nms import non_max_suppression  # newer versions
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression  # older versions
            dev = next(trainer.model.parameters()).device
            net = trainer.model
            was_training = net.training
            net.eval()
            with torch.no_grad():
                out = net(imgs[0:1].to(dev))
            if was_training:
                net.train()
            y = out[0] if isinstance(out, (tuple, list)) else out
            det = non_max_suppression(y.float(), conf_thres=0.25, iou_thres=0.45)[0]

            # RAW classifier output: per-class best score straight from the
            # head tensor [1, 4+nc, N] — before any NMS filtering.
            classes = []
            try:
                if hasattr(y, "shape") and y.ndim == 3 and y.shape[1] > 4:
                    cls_scores = y[0, 4:, :]
                    best = cls_scores.max(dim=1).values          # best anchor per class
                    k = min(8, best.numel())
                    top = torch.topk(best, k=k)
                    for v, i in zip(top.values.tolist(), top.indices.tolist()):
                        classes.append({
                            "class_id": int(i),
                            "name": self.class_names.get(int(i), str(int(i))),
                            "prob": round(float(v), 4),
                        })
            except Exception:
                pass

            if det is not None and len(det):
                sx, sy = arr.shape[1] / w, arr.shape[0] / h
                for *xyxy, conf, cid in det.cpu().numpy():
                    x1, y1, x2, y2 = xyxy
                    pred.append({
                        "class_id": int(cid),
                        "name": self.class_names.get(int(cid), str(int(cid))),
                        "confidence": round(float(conf), 3),
                        "bbox": [round(x1 * sx, 1), round(y1 * sy, 1),
                                 round((x2 - x1) * sx, 1), round((y2 - y1) * sy, 1)],
                    })
        except Exception as e:
            print(f"[viz] prediction capture error: {e}", flush=True)

        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        import base64

        # The three REAL input channels (R, G, B) of this exact augmented image,
        # shown as separate maps like CNN-Explainer's input column.
        input_channels = []
        try:
            b_ch, g_ch, r_ch = cv2.split(arr)
            for ch in (r_ch, g_ch, b_ch):
                okc, bufc = cv2.imencode(".jpg", ch, [cv2.IMWRITE_JPEG_QUALITY, 68])
                if okc:
                    input_channels.append(base64.b64encode(bufc.tobytes()).decode("ascii"))
        except Exception:
            pass

        frame = {
            "image_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
            "input_channels": input_channels,
            "gt": gt,
            "pred": pred,
            "classes": classes,   # raw per-class scores from the head (pre-NMS)
            "epoch": self.metrics.get("epoch", 0),
            "batch": self._batch_counter,
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
        }
        with self.lock:
            self.frame = frame
            self.frame_seq += 1
            print(f"[viz] captured frame epoch={frame['epoch']} batch={frame['batch']} gt={len(gt)} pred={len(pred)}", flush=True)

    def _grad_thumbs(self, idx, shape, channels: int = 2):
        """Render the latest real backward gradient (dL/doutput) of a layer."""
        import torch.nn.functional as F
        import torch
        g = self._grad_activations.get(idx)
        if g is None:
            return []
        out = []
        try:
            t = g[0] if g.ndim == 4 else g
            t = t.abs().detach().float().cpu()
            t = F.adaptive_avg_pool2d(t.unsqueeze(0), (PIPELINE_SIZE * 2, PIPELINE_SIZE * 2))[0]
            for c in t[:channels]:
                b64 = _thumb(c.numpy(), PIPELINE_SIZE)
                if b64:
                    out.append(b64)
        except Exception:
            pass
        return out

    def _package_pipeline(self):
        """Build the CNN-Explainer-style stage list from the latest real
        activations of EVERY hooked layer. Layer 0 is decomposed into its
        real sub-stages (raw conv -> batchnorm -> activation) using extra
        hooks on the inner nn.Conv2d / nn.BatchNorm2d modules."""
        stages = []

        def make_stage(key, meta, act):
            try:
                maps, is_multiscale = _tensor_maps(act)
                thumbs = []
                vmin = vmax = 0.0
                if is_multiscale:
                    for m in maps:
                        b64, lo, hi = _thumb_div(m, PIPELINE_SIZE)
                        if b64:
                            thumbs.append(b64)
                            vmax = max(vmax, abs(lo), abs(hi))
                else:
                    for c in maps[:PIPELINE_CHANNELS]:
                        b64, lo, hi = _thumb_div(c, PIPELINE_SIZE)
                        if b64:
                            thumbs.append(b64)
                            vmax = max(vmax, abs(hi))
                if not thumbs:
                    return None
                shape = []
                try:
                    t = act
                    if isinstance(t, (list, tuple)):
                        t = t[0]
                    elif isinstance(t, dict):
                        t = next(iter(t.values()))
                    shape = [int(x) for x in t.shape][:4]
                except Exception:
                    pass
                return {
                    "index": key,
                    "name": meta.get("name", str(key)),
                    "params": meta.get("params", 0),
                    "is_head": meta.get("is_head", False),
                    "from": meta.get("from", []),
                    "activation": meta.get("activation"),
                    "conv": meta.get("conv"),
                    "shape": shape,
                    "thumbs": thumbs,
                    "scale": {"vmin": -round(vmax, 3), "vmax": round(vmax, 3)},
                    # Real BACKWARD-pass gradients (dL/doutput) as heatmaps
                    "bwd": self._grad_thumbs(key if isinstance(key, int) else int(str(key)[0]),
                                             shape) if isinstance(key, int) else [],
                }
            except Exception:
                return None

        for idx in sorted(self._activations.keys()):
            if not isinstance(idx, int):
                continue
            stage = make_stage(idx, self.layer_meta.get(idx, {"name": f"Layer {idx}"}), self._activations[idx])
            if stage:
                stages.append(stage)
                # Decompose the first conv into its real inner steps.
                if idx == 0:
                    for sub_key, label in (("0c", "conv 0 (raw)"), ("0b", "batchnorm 0")):
                        sub_act = self._sub_activations.get(sub_key)
                        if sub_act is None:
                            continue
                        sub = make_stage(sub_key, {"name": label, "params": 0, "from": []}, sub_act)
                        if sub:
                            stages.append(sub)
        if stages:
            with self.lock:
                self.pipeline = stages
                self.pipeline_seq += 1

    # ------------------------------------------------------------------
    # Snapshot helpers for websocket streaming / REST inspection
    # ------------------------------------------------------------------
    def snapshot_metrics(self) -> dict:
        with self.lock:
            return dict(self.metrics)

    def snapshot_frame_if_new(self, last_seq: int):
        with self.lock:
            if self.frame is not None and self.frame_seq != last_seq:
                return self.frame_seq, dict(self.frame)
        return last_seq, None

    def snapshot_pipeline_if_new(self, last_seq: int):
        with self.lock:
            if self.pipeline is not None and self.pipeline_seq != last_seq:
                return self.pipeline_seq, [dict(s) for s in self.pipeline]
        return last_seq, None

    def snapshot_grad_norms(self) -> dict:
        with self.lock:
            return dict(self.grad_norms)

    def snapshot_grad_seq(self) -> int:
        with self.lock:
            return self.grad_seq

    def snapshot_status(self) -> str:
        with self.lock:
            return self.status


def _gray_thumb(fmap: np.ndarray, size: int) -> str | None:
    """Grayscale render (for showing raw kernel weights)."""
    import cv2, base64
    try:
        f = np.asarray(fmap, dtype=np.float32)
        mn, mx = float(f.min()), float(f.max())
        if mx - mn > 1e-9:
            f = (f - mn) / (mx - mn)
        else:
            f = np.zeros_like(f)
        small = cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".jpg", (small * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def layer_detail(vis: TrainingVisualizer, layer_index: int, channels: int = DETAIL_CHANNELS, size: int = DETAIL_SIZE) -> dict:
    """Deep inspection of one layer's latest real activation (REST endpoint)."""
    act = vis._activations.get(layer_index)
    meta = vis.layer_meta.get(layer_index, {})
    if act is None and not meta:
        return {"exists": False}

    import cv2, base64
    import torch
    import torch.nn.functional as F

    chans = []
    shape = []
    name = meta.get("name", f"Layer {layer_index}")
    filters = []
    bias_values = []

    def pool(t):
        t = t.detach().float().cpu()
        t = F.adaptive_avg_pool2d(t.unsqueeze(0), (size * 2, size * 2))[0]
        return t.numpy()

    try:
        if isinstance(act, dict):
            tensors = [v for v in act.values() if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 3][:channels]
            for s in tensors:
                m = s[0] if s.ndim == 4 else s
                b64 = _thumb(m.mean(dim=0).detach().float().cpu().numpy(), size, 75)
                if b64:
                    chans.append({"label": f"scale {len(chans)}", "b64": b64})
                shape = [int(x) for x in s.shape]
        elif isinstance(act, (list, tuple)):
            for s in [a for a in act if hasattr(a, "shape")][:channels]:
                m = s[0] if s.ndim == 4 else s
                b64 = _thumb(m.mean(dim=0).detach().float().cpu().numpy(), size, 75)
                if b64:
                    chans.append({"label": f"scale {len(chans)}", "b64": b64})
                shape = [int(x) for x in s.shape]
        elif act is not None and hasattr(act, "shape"):
            t = act[0] if act.ndim == 4 else act
            shape = [int(x) for x in t.shape]
            maps = pool(t)
            step = max(1, maps.shape[0] // channels)
            sel = list(range(0, maps.shape[0], step))[:channels]
            for ci in sel:
                b64 = _thumb(maps[ci], size, 75)
                if b64:
                    chans.append({"label": f"ch {ci}", "b64": b64})
    except Exception as e:
        pass

    # ---- REAL FILTERS: actual learned kernel weights of this layer ----
    try:
        module = vis._modules_by_idx.get(layer_index)
        conv = getattr(module, "conv", None) if module is not None else None
        target = conv if conv is not None else module
        w = getattr(target, "weight", None) if target is not None else None
        if w is not None and hasattr(w, "shape") and len(w.shape) == 4:
            n_out = min(6, int(w.shape[0]))
            for oi in range(n_out):
                k = w[oi].detach().float().cpu()          # [in, kh, kw]
                km = k.mean(dim=0).numpy()                 # average across input chs
                b64 = _gray_thumb(km, 64)
                if b64:
                    filters.append({"label": f"filter {oi}", "b64": b64})
            b = getattr(target, "bias", None)
            if b is not None:
                bias_values = [round(float(x), 4) for x in b.detach().flatten()[:8].tolist()]
    except Exception as e:
        print(f"[viz] filter extraction error: {e}", flush=True)

    return {
        "exists": True,
        "index": int(layer_index),
        "name": name,
        "shape": shape,
        "meta": meta,
        "channels": chans,           # real forward activations (diverging)
        "filters": filters,          # real learned kernels (grayscale)
        "bias_values": bias_values,  # first real bias numbers
        "grads": _grad_detail_maps(vis, layer_index),
    }


def _grad_detail_maps(vis: "TrainingVisualizer", layer_index: int, channels: int = 4) -> list:
    """Real backward-pass gradients (dL/doutput) of this layer, diverging maps."""
    g = vis._grad_activations.get(layer_index)
    if g is None:
        return []
    import torch.nn.functional as F
    out = []
    try:
        t = g[0] if g.ndim == 4 else g
        t = t.abs().detach().float().cpu()
        t = F.adaptive_avg_pool2d(t.unsqueeze(0), (DETAIL_SIZE * 2, DETAIL_SIZE * 2))[0]
        step = max(1, t.shape[0] // channels)
        for ci in list(range(0, t.shape[0], step))[:channels]:
            b64 = _thumb(t[ci].numpy(), DETAIL_SIZE, 75)
            if b64:
                out.append({"label": f"∂L/∂out ch{ci}", "b64": b64})
    except Exception:
        pass
    return out
