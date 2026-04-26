"""
Vision Engine — Facial emotion detection and confidence scoring.

Ported from the Streamlit app.py.  This module is framework-agnostic:
it takes NumPy arrays in and returns Python dicts out.  No Django or
Streamlit imports live here.
"""

import base64
import threading
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Labels — keep aligned with training order
# ---------------------------------------------------------------------------
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------------------------------------------------------------------
# Thread-safe singleton holders
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_model_instance = None
_detector_instance = None


def build_model(input_shape=(48, 48, 1), num_classes=len(EMOTION_LABELS)):
    """
    Define the emotion model architecture used by a weights-only checkpoint.

    If your .weights.h5 file was trained with a different architecture, update
    this function to match that exact training model before calling load_weights.
    Full saved .h5 models are loaded directly and do not need this architecture.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.30),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.50),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def load_emotion_model(model_path: str):
    """
    Load either a full saved Keras .h5 model or a weights-only .weights.h5 file.
    Thread-safe singleton — loaded once and reused across WebSocket connections.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:
            return _model_instance

        model_path = Path(model_path)

        if model_path.name.endswith(".weights.h5"):
            model = build_model()
            model.load_weights(str(model_path))
        else:
            try:
                model = tf.keras.models.load_model(str(model_path), compile=False)
            except Exception:
                model = build_model()
                model.load_weights(str(model_path))

        _model_instance = model
        return _model_instance


def load_face_detector():
    """Load the Haar cascade face detector (singleton)."""
    global _detector_instance
    if _detector_instance is not None:
        return _detector_instance

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    _detector_instance = detector
    return _detector_instance


def preprocess_face(gray_frame, face_box):
    """Crop a face ROI from a grayscale NumPy frame and prepare it for Keras."""
    x, y, w, h = face_box
    roi = gray_frame[y : y + h, x : x + w]
    roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))
    return roi


def confidence_score_from_emotion(emotion: str, probability: float) -> int:
    """
    Map an emotion to a 1-10 confidence score and use model probability to vary
    inside the requested baseline band.
    """
    bands = {
        "Happy": (8, 10),
        "Neutral": (8, 10),
        "Surprise": (6, 7),
        "Sad": (4, 5),
        "Angry": (4, 5),
        "Fear": (1, 3),
        "Disgust": (1, 3),
    }
    low, high = bands.get(emotion, (1, 5))
    probability = float(np.clip(probability, 0.0, 1.0))
    score = low + round(probability * (high - low))
    return int(np.clip(score, 1, 10))


def predict_emotion(model, gray_frame, face_box):
    """Run emotion inference on a single face ROI."""
    face_tensor = preprocess_face(gray_frame, face_box)
    predictions = model.predict(face_tensor, verbose=0)[0]
    if len(predictions) != len(EMOTION_LABELS):
        raise ValueError(
            f"Model returned {len(predictions)} classes, but EMOTION_LABELS has "
            f"{len(EMOTION_LABELS)} labels. Update EMOTION_LABELS to match training."
        )
    emotion_index = int(np.argmax(predictions))
    probability = float(predictions[emotion_index])
    emotion = EMOTION_LABELS[emotion_index]
    score = confidence_score_from_emotion(emotion, probability)
    return emotion, probability, score


def annotate_face(frame_bgr, face_box, emotion, probability, score):
    """Draw bounding box and label on the frame."""
    x, y, w, h = face_box
    color = (
        (40, 220, 80) if score >= 8 else (40, 180, 255) if score >= 5 else (60, 80, 255)
    )
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    label = f"{emotion} {probability * 100:.1f}% | Confidence {score}/10"
    cv2.putText(
        frame_bgr,
        label,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame_bgr


def process_video_frame(jpeg_bytes: bytes, model, face_detector) -> dict:
    """
    Full pipeline for one video frame.

    Accepts raw JPEG bytes from the browser, returns a dict with results
    and the annotated frame as a base64 JPEG string.
    """
    # Decode JPEG bytes → OpenCV BGR image
    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return {
            "emotion": "Error",
            "probability": 0.0,
            "score": 0,
            "faces_detected": 0,
            "annotated_frame": "",
        }

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    primary_emotion = "No face detected"
    primary_probability = 0.0
    primary_score = 0

    if len(faces) > 0:
        # Use the largest face for the primary score
        primary_face = max(faces, key=lambda box: box[2] * box[3])
        primary_emotion, primary_probability, primary_score = predict_emotion(
            model, gray, primary_face
        )

        # Annotate all faces
        for face_box in faces:
            if np.array_equal(face_box, primary_face):
                emotion, prob, sc = primary_emotion, primary_probability, primary_score
            else:
                emotion, prob, sc = predict_emotion(model, gray, face_box)
            annotate_face(frame_bgr, face_box, emotion, prob, sc)

    # Encode annotated frame → JPEG → base64
    _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "emotion": primary_emotion,
        "probability": round(primary_probability, 4),
        "score": primary_score,
        "faces_detected": len(faces),
        "annotated_frame": annotated_b64,
    }
