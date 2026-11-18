"""
Gradio demo: upload a leaf photo -> top-3 diseases + a Grad-CAM heatmap showing where the model looked.

Needs model.keras and labels.json in this folder (both are produced by the Kaggle notebook,
inside outputs.zip -> app/).

    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr
import keras
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).parent
META = json.loads((HERE / "labels.json").read_text())
CLASSES = META["classes"]
IMG = META["image_size"]
MODEL = keras.models.load_model(HERE / "model.keras")


def _last_conv(model):
    try:
        return model.get_layer("top_activation")
    except ValueError:
        return next(l for l in reversed(model.layers) if len(l.output.shape) == 4)


GRAD_MODEL = keras.Model(MODEL.input, [_last_conv(MODEL).output, MODEL.output])


def label(i: int) -> str:
    c = CLASSES[i]
    return f"{c['plant']}: healthy" if c["healthy"] else f"{c['plant']}: {c['disease']}"


def gradcam(x: np.ndarray):
    with tf.GradientTape() as tape:
        conv, preds = GRAD_MODEL(x, training=False)
        idx = int(tf.argmax(preds[0]))
        target = preds[:, idx]
    grads = tf.cast(tape.gradient(target, conv), tf.float32)
    conv = tf.cast(conv, tf.float32)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights[0], axis=-1))
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    return cam, preds[0].numpy()


def predict(image: Image.Image):
    if image is None:
        return {}, None, "Upload a photo of a single leaf."
    rgb = image.convert("RGB")
    small = rgb.resize((IMG, IMG), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32)[None]          # the model rescales internally (0-255 input)
    cam, probs = gradcam(x)

    heat = Image.fromarray(np.uint8(cam * 255)).resize(rgb.size, Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    overlay = (0.55 * np.asarray(rgb, dtype=np.float32) / 255.0 + 0.45 * colored)
    overlay = Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))

    top = np.argsort(-probs)[:3]
    scores = {label(int(i)): float(probs[i]) for i in top}
    best = CLASSES[int(top[0])]
    if probs[top[0]] < 0.5:
        note = "Low confidence: retake the photo closer to one leaf, in daylight, with the affected area visible."
    elif best["healthy"]:
        note = "The leaf looks healthy."
    else:
        note = f"Likely {best['disease']} on {best['plant']}. Confirm with a local agriculture officer before treating."
    return scores, overlay, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Leaf photo"),
    outputs=[gr.Label(num_top_classes=3, label="Prediction"),
             gr.Image(label="Where the model looked (Grad-CAM)"),
             gr.Textbox(label="Note")],
    title="Field-Ready Plant Disease Detection",
    description="EfficientNetV2-B0 trained on PlantVillage and fine-tuned on real field photos (PlantDoc). "
                "38 classes across 14 crops. A study project, not a substitute for expert diagnosis.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
