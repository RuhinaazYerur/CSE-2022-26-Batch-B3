import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array




app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = os.path.join("resnet_rice_model.h5")  # change if your filename differs
IMG_SIZE = (224, 224)

# IMPORTANT: must match your training class order (from your screenshot)
CLASS_NAMES = ["fully_cooked", "raw", "semi_cooked"]


# -----------------------------
# Load model once
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH,compile=False)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXT

def predict_image(file_path: str):
    img = load_img(file_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = float(probs[idx])
    return label, conf, probs

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", title="Rice Classification")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", title="About")

@app.route("/performance", methods=["GET", "POST"])
def performance():
    return render_template("performance.html")

@app.route("/dataset", methods=["GET"])
def dataset():
    return render_template("dataset.html", title="Dataset")
    
@app.route("/know-more")
def know_more():
    return render_template("know_more.html")

@app.route("/team", methods=["GET"])
def team():
    return render_template("team.html", title="Team")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", title="Rice Classification", error="No file selected.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", title="Rice Classification", error="Please choose an image.")

    if not allowed_file(file.filename):
        return render_template("index.html", title="Rice Classification", error="Upload JPG / PNG / WEBP only.")

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    label, conf, probs = predict_image(save_path)

    # For UI: show confidence bar + show all class probabilities
    prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return render_template(
        "result.html",
        title="Result",
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
        label=label,
        conf=conf,
        prob_map=prob_map
    )

if __name__ == "__main__":
    app.run(debug=True)
