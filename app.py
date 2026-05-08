from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
from db import save_prediction, get_all_predictions

app = Flask(__name__)
CORS(app)

print("Loading model...")

# LOAD MODEL
model = load_model("model/model.h5")

# LOAD LABELS
with open("model/class_labels.json") as f:
    class_labels = json.load(f)

# LOAD TREATMENTS
with open("treatments.json") as f:
    treatments = json.load(f)

print("Model ready!")

# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")

# IMAGE PREPROCESSING
def preprocess_image(file):

    img = Image.open(file).convert("RGB")

    img = img.resize((224, 224))

    arr = np.array(img) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:

        # PREPROCESS IMAGE
        arr = preprocess_image(file)

        # MODEL PREDICTION
        preds = model.predict(arr)[0]

        # CONFIDENCE
        conf = float(np.max(preds)) * 100

        # PREDICTED INDEX
        idx = int(np.argmax(preds))

        # TOP 3 DEBUG
        top3 = preds.argsort()[-3:][::-1]

        print("\nTop Predictions:")

        for i in top3:

            label = class_labels.get(str(i), "Unknown")

            score = float(preds[i]) * 100

            print(f"{label} --> {score:.2f}%")

        # LOW CONFIDENCE CHECK
        if conf < 50:

            return jsonify({

                "disease": "Unknown Disease",

                "confidence": f"{conf:.2f}%",

                "treatment":
                "Image unclear. Please upload a clearer leaf image.",

                "status": "Uncertain Prediction",

                "is_healthy": False

            })

        # DISEASE LABEL
        disease = class_labels.get(str(idx), "Unknown")

        # CLEAN NAME
        clean_disease = disease.replace(
            "___",
            " "
        ).replace(
            "_",
            " "
        ).strip()

        # TREATMENT
        treatment = treatments.get(

            clean_disease,

            "Consult an agricultural expert."

        )

        # HEALTH STATUS
        is_healthy = "healthy" in clean_disease.lower()

        # FINAL RESULT
        result = {

            "disease": clean_disease,

            "confidence": f"{conf:.2f}%",

            "treatment": treatment,

            "is_healthy": is_healthy,

            "status":
            "Healthy" if is_healthy
            else "Disease Detected"
        }

        # SAVE HISTORY
        save_prediction({

            "disease": clean_disease,

            "confidence": f"{conf:.2f}%",

            "treatment": treatment,

            "filename": file.filename

        })

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# HISTORY ROUTE
@app.route("/history", methods=["GET"])
def history():

    return jsonify(get_all_predictions())

# START SERVER
if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )