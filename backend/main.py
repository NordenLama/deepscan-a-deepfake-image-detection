"""FastAPI application for whole-image HOG/LBP/FFT deepfake image classification."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

if __package__:
    from .feature_extractor import (
        decode_image,
        features_from_image,
        prepare_image,
        texture_hotspots,
    )
else:
    from feature_extractor import (
        decode_image,
        features_from_image,
        prepare_image,
        texture_hotspots,
    )

logger = logging.getLogger("deepscan")
MODEL_DIRECTORY = Path(__file__).resolve().parent / "saved_models"
MODEL_FILES = {
    "svm": "svm.joblib",
    "random_forest": "random_forest.joblib",
    "ann": "ann.joblib",
    "decision_tree": "decision_tree.joblib",
    "ensemble": "ensemble.joblib",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_VERDICT_PROBABILITY = 0.65


def load_saved_models() -> dict[str, dict[str, Any]]:
    """Load available model bundles, reporting corrupt artifacts explicitly."""
    import joblib

    models: dict[str, dict[str, Any]] = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIRECTORY / filename
        if not model_path.is_file():
            continue
        try:
            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict) or "estimator" not in bundle:
                logger.warning("Model artifact has an invalid format: %s", model_path)
                continue
            models[model_name] = bundle
        except Exception as error:
            logger.error("Failed to load model %s: %s", model_name, error)

    if not models:
        logger.warning(
            "No trained models found in %s. Run backend/train_models.py before predicting.",
            MODEL_DIRECTORY,
        )
    else:
        logger.info("Loaded detection models: %s", ", ".join(sorted(models)))
    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_saved_models()
    yield


app = FastAPI(
    title="Deepfake Image Detection API",
    description="Whole-image HOG/LBP/FFT features with supervised scikit-learn classifiers.",
    version="1.0.1",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def analyze_image(image_bytes: bytes, estimator: Any) -> dict[str, Any]:
    """Extract features and abstain when the classifier probability is too weak."""
    image = decode_image(image_bytes)
    features = features_from_image(image).reshape(1, -1)
    prediction = int(estimator.predict(features)[0])

    if not hasattr(estimator, "predict_proba"):
        raise ValueError("The selected model does not provide class probabilities.")
    probabilities = estimator.predict_proba(features)[0]
    classes = list(estimator.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError("The selected model is missing the real/fake class mapping.")

    real_probability = float(probabilities[classes.index(0)])
    fake_probability = float(probabilities[classes.index(1)])
    confidence = max(real_probability, fake_probability)
    is_uncertain = confidence < MIN_VERDICT_PROBABILITY
    is_fake = None if is_uncertain else prediction == 1

    # Generate texture hotspots on the preprocessed image view
    processed_image = prepare_image(image)

    return {
        "verdict": (
            "Uncertain"
            if is_uncertain
            else "AI Generated / Manipulated"
            if is_fake
            else "Real (Authentic)"
        ),
        "isFake": is_fake,
        "isUncertain": is_uncertain,
        "confidence": round(confidence * 100, 1),
        "real_pct": round(real_probability * 100, 1),
        "fake_pct": round(fake_probability * 100, 1),
        "model": "",
        "heatmap_coordinates": texture_hotspots(processed_image),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    # Refresh model state dynamically if models were trained after boot
    loaded_models = getattr(app.state, "models", {})
    if not loaded_models:
        app.state.models = load_saved_models()
        loaded_models = app.state.models

    return {
        "status": "ok",
        "loaded_models": sorted(loaded_models.keys()),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("svm"),
) -> dict[str, Any]:
    if model not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Choose one of: {', '.join(MODEL_FILES)}.",
        )

    # Ensure models are loaded even if startup check missed them before training
    models = getattr(app.state, "models", {})
    if model not in models:
        app.state.models = load_saved_models()
        models = app.state.models

    if model not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is not trained. Run backend/train_models.py first.",
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image uploads are limited to 10 MB.")

    started_at = perf_counter()
    try:
        result = await run_in_threadpool(analyze_image, image_bytes, models[model]["estimator"])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    result["model"] = model
    result["latency_ms"] = round((perf_counter() - started_at) * 1000, 1)
    return result