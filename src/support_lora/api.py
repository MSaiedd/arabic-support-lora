"""Browser comparison API. Inference never triggers training or downloads per request."""

import os
import time
from pathlib import Path
from typing import Literal

import joblib
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .baseline import predict
from .security import make_auth


class PredictionRequest(BaseModel):
    text: str = Field(min_length=3, max_length=1500)
    variant: Literal["baseline", "base", "lora"] = "baseline"


def create_app(tokens=None, baseline_path=None, causal=None):
    auth = make_auth(tokens)
    path = Path(baseline_path or os.environ.get("BASELINE_PATH", "artifacts/baseline.joblib"))
    # Only load trusted local artifacts: joblib is not an untrusted upload format.
    baseline = joblib.load(path) if path.exists() else None
    if causal is None and os.environ.get("MODEL_ID"):
        from .inference import CausalPredictor

        causal = CausalPredictor(
            os.environ["MODEL_ID"],
            os.environ.get("ADAPTER_PATH"),
            os.environ.get("MODEL_REVISION", "main"),
        )
    app = FastAPI(title="Arabic-English Support LoRA", version="0.1.0")
    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "baseline_ready": baseline is not None,
            "base_ready": causal is not None,
            "lora_ready": causal is not None and causal.has_adapter,
        }

    @app.post("/predict")
    def prediction(body: PredictionRequest, principal=Depends(auth)):
        started = time.perf_counter()
        if body.variant == "baseline":
            if baseline is None:
                raise HTTPException(
                    503, "Train the local baseline first: python -m support_lora.baseline"
                )
            result = predict(baseline, body.text)
        else:
            if causal is None or (body.variant == "lora" and not causal.has_adapter):
                raise HTTPException(
                    503, "Configure MODEL_ID and ADAPTER_PATH before starting the server"
                )
            result = causal.predict(body.text, body.variant)
        return {**result, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    return app
