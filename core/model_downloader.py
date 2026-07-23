"""
model_downloader.py

Downloads MediaPipe's official pre-trained task model files on first
run if they're not already present locally. These are Google's
published model bundles for the MediaPipe Tasks API (the API that
replaced the older mp.solutions.* interface in recent MediaPipe
releases).
"""

import os
import urllib.request

MODEL_URLS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def ensure_model(filename: str) -> str:
    """
    Ensures the given MediaPipe model file exists in the models/
    directory, downloading it from Google's model zoo if missing.
    Returns the local file path.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    if filename not in MODEL_URLS:
        raise ValueError(f"No known download URL for model file: {filename}")

    print(f"[model_downloader] Downloading {filename} (first run only)...")
    url = MODEL_URLS[filename]
    urllib.request.urlretrieve(url, local_path)
    print(f"[model_downloader] Saved to {local_path}")
    return local_path