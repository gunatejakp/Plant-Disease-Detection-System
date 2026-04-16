import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from disease_info import disease_info

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = load_model("model/plant_model.h5")

# Load class names
with open("model/class_names.json") as f:
    class_names = json.load(f)

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    if file:
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = preprocess(path)
        preds = model.predict(img)[0]

        idx = np.argmax(preds)
        confidence = float(np.max(preds))

        predicted_class = class_names[idx]

        info = disease_info.get(predicted_class, {
            "cause": "Unknown",
            "cure": "Consult expert"
        })

        return render_template("result.html",
                               image_path=path,
                               disease=predicted_class,
                               cause=info["cause"],
                               cure=info["cure"],
                               confidence=round(confidence*100, 2))

    return "Error"

if __name__ == "__main__":
    app.run(debug=True)