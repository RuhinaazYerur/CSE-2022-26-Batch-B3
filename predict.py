import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "resnet_rice_model.h5"   # or .keras
IMG_SIZE = (224, 224)

# Class names must match dataset folders
CLASS_NAMES = ["fully_cooked","raw", "semi_cooked"]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    index = np.argmax(preds)

    return CLASS_NAMES[index], preds[index]

if __name__ == "__main__":
    image_path = "test.jpg"   # put any image path here
    label, confidence = predict_image(image_path)
    print("Prediction:", label)
    print("Confidence:", round(confidence*100, 2), "%")
