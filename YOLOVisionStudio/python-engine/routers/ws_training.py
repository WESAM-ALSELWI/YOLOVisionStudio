import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from routers.training import jobs_store

router = APIRouter(tags=["LiveTraining"])

SEND_INTERVAL = 0.25  # seconds between event pushes


@router.websocket("/ws/training/{job_id}")
async def training_ws(websocket: WebSocket, job_id: str):
    """Streams REAL data from the running YOLO training process:
    per-batch losses/lr, live input frames with ground-truth vs predicted
    boxes, and intermediate layer activations (throttled captures)."""
    await websocket.accept()

    state = jobs_store.get(job_id)
    if state is None:
        await websocket.close(code=4404)
        return

    vis = getattr(state, "visualizer", None)
    last_frame_seq = -1
    last_pipeline_seq = -1
    last_core_seq = -1
    last_status = None

    try:
        while True:
            payload = {"type": "tick", "job_status": state.status}

            if vis is not None:
                payload["metrics"] = vis.snapshot_metrics()
                payload["class_names"] = vis.class_names
                # Real backprop strength per layer (weight-grad L2 norms)
                payload["grad_norms"] = vis.snapshot_grad_norms()

                last_frame_seq, frame = vis.snapshot_frame_if_new(last_frame_seq)
                if frame is not None:
                    payload["frame"] = frame

                # CNN-Explainer-style snapshot: every real layer's activations
                last_pipeline_seq, pipeline = vis.snapshot_pipeline_if_new(last_pipeline_seq)
                if pipeline is not None:
                    payload["pipeline"] = pipeline
                    payload["pipeline_seq"] = last_pipeline_seq

                if hasattr(vis, "snapshot_core_stages_if_new"):
                    last_core_seq, core_stages = vis.snapshot_core_stages_if_new(last_core_seq)
                    if core_stages is not None:
                        payload["core_stages"] = core_stages

            status = vis.snapshot_status() if vis else state.status
            if status != last_status:
                payload["status"] = status
                last_status = status
                if state.error_message:
                    payload["error"] = state.error_message

            await websocket.send_json(payload)

            finished = status in ("completed", "failed", "cancelled")
            if finished and websocket.client_state.name == "CONNECTED":
                # Give clients a moment to receive the final tick before closing.
                await asyncio.sleep(1.0)
                await websocket.close(code=1000)
                break

            await asyncio.sleep(SEND_INTERVAL)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
