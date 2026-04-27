from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blazar_classification_app.inference import FEATURE_COLUMNS, metadata_payload, predict_manual, predict_upload


class ManualRequest(BaseModel):
    alpha: float = 0.1
    features: dict[str, float]


app = FastAPI(title="Blazar Classification API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    return metadata_payload()


@app.post("/predict/manual")
def predict_manual_route(request: ManualRequest) -> dict:
    missing = [column for column in FEATURE_COLUMNS if column not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing feature values: {', '.join(missing)}")
    try:
        return predict_manual(request.features, alpha=request.alpha)
    except Exception as exc:  # pragma: no cover - API wrapper
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/upload")
async def predict_upload_route(
    file: UploadFile = File(...),
    alpha: float = Form(0.1),
    input_mode: str = Form("auto"),
) -> dict:
    try:
        payload = await file.read()
        return predict_upload(file.filename or "uploaded.csv", payload, input_mode=input_mode, alpha=alpha)
    except Exception as exc:  # pragma: no cover - API wrapper
        raise HTTPException(status_code=400, detail=str(exc)) from exc
