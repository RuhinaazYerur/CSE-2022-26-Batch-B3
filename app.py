import os
import gc
import uuid
import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = "resnet_rice_model.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["fully_cooked", "raw", "semi_cooked"]
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

model = None
tf = None

def init_tf():
    global tf
    if tf is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tensorflow_lib
        tensorflow_lib.config.threading.set_intra_op_parallelism_threads(1)
        tensorflow_lib.config.threading.set_inter_op_parallelism_threads(1)
        tf = tensorflow_lib
    return tf

def get_model():
    global model
    if model is None:
        tensorflow_lib = init_tf()
        model = tensorflow_lib.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXT

def predict_image(file_path: str):
    tensorflow_lib = init_tf()
    current_model = get_model()

    img = Image.open(file_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)
    arr = tensorflow_lib.keras.applications.resnet50.preprocess_input(arr)

    probs = current_model.predict(arr, verbose=0)[0]

    del img
    del arr
    gc.collect()

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
    return render_template("performance.html", title="Performance")

@app.route("/dataset", methods=["GET"])
def dataset():
    return render_template("dataset.html", title="Dataset")

@app.route("/know-more", methods=["GET"])
def know_more():
    return render_template("know_more.html", title="Know More")

@app.route("/team", methods=["GET"])
def team():
    return render_template("team.html", title="Team")

@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html", title="Upload")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return render_template("index.html", title="Rice Classification", error="No file selected.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", title="Rice Classification", error="Please choose an image.")

        if not allowed_file(file.filename):
            return render_template("index.html", title="Rice Classification", error="Upload JPG / JPEG / PNG / WEBP only.")

        original_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        label, conf, probs = predict_image(save_path)
        prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return render_template(
            "result.html",
            title="Result",
            image_url=url_for("static", filename=f"uploads/{unique_name}"),
            label=label,
            conf=conf,
            prob_map=prob_map
        )

    except Exception as e:
        return render_template("index.html", title="Rice Classification", error=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)