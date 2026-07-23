"""
expression_classifier.py

Runs a FER2013-trained TFLite model on a cropped face image to classify
visible facial expression. Outputs are treated as probabilistic,
visible-cue labels only (e.g. "face consistent with sadness"), never
as certain statements about inner emotional state -- per the concept
paper's transparency requirement.

NOTE: This module expects a trained model file at:
    models/fer2013_model.tflite

You'll need to supply this yourself -- e.g. train one on the FER2013
dataset (Kaggle) with a standard CNN architecture, then convert to
.tflite with tf.lite.TFLiteConverter. This file only handles loading
and running inference, not training.
"""

import numpy as np
import cv2
import tensorflow as tf

# Standard FER2013 label order -- verify this matches your trained model's
# output layer order, since FER2013 dataset label ordering can vary
# depending on the training pipeline used.
EXPRESSION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

MODEL_INPUT_SIZE = (48, 48)  # FER2013 standard input size, grayscale


class ExpressionClassifier:
    def __init__(self, model_path: str = "models/fer2013_model.tflite"):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_loaded = True
        except (ValueError, OSError):
            # Model file missing/invalid -- degrade gracefully so the rest
            # of the pipeline (camera, face/pose detection, UI) can still
            # be demoed and tested without a trained model in hand yet.
            self.model_loaded = False
            print(
                f"[ExpressionClassifier] WARNING: could not load model at "
                f"'{model_path}'. Running in stub mode -- classify() will "
                f"return 'neutral' with 0 confidence until a real model "
                f"is supplied."
            )

    def preprocess(self, face_bgr):
        """Converts a cropped BGR face image to the model's expected input."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, MODEL_INPUT_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        # Shape: (1, 48, 48, 1) -- batch, height, width, channels
        return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

    def classify(self, face_bgr):
        """
        Returns (label: str, confidence: float).
        label is one of EXPRESSION_LABELS, confidence is 0.0-1.0.
        """
        if not self.model_loaded:
            return "neutral", 0.0

        if face_bgr is None or face_bgr.size == 0:
            return "neutral", 0.0

        input_tensor = self.preprocess(face_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        best_idx = int(np.argmax(output))
        confidence = float(output[best_idx])
        return EXPRESSION_LABELS[best_idx], confidence
