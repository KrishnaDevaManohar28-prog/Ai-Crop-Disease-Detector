from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from db import db, Prediction

import numpy as np
import json
import os

app = Flask(__name__)

# =========================
# DATABASE CONFIGURATION
# =========================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()

# =========================
# UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = "uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================
# LOAD AI MODEL
# ========================
model = load_model(
    "model/model.h5",
    compile=False
)

# =========================
# LOAD CLASS LABELS
# =========================
with open("model/class_labels.json", "r") as f:
    class_labels = json.load(f)

# =========================
# LOAD TREATMENTS
# =========================
with open("treatments.json", "r") as f:
    treatments = json.load(f)

# =========================
# HOME PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    uploaded_image = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                result="No image uploaded"
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                result="No file selected"
            )

        # =========================
        # SAVE IMAGE
        # =========================
        filename = file.filename

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        uploaded_image = filepath

        # =========================
        # IMAGE PREPROCESSING
        # =========================
        img = image.load_img(
            filepath,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0

        # =========================
        # AI PREDICTION
        # =========================
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction)

        confidence_score = round(
            float(np.max(prediction)) * 100,
            2
        )

        predicted_class = class_labels[str(predicted_index)]

        # =========================
        # GET TREATMENT
        # =========================
        treatment = treatments.get(
            predicted_class,
            "No treatment available"
        )

        # =========================
        # SAVE TO DATABASE
        # =========================
        new_prediction = Prediction(
            image_name=filename,
            disease=predicted_class,
            confidence=str(confidence_score),
            remedy=treatment
        )

        db.session.add(new_prediction)
        db.session.commit()

        # =========================
        # RESULT
        # =========================
        result = {
            "disease": predicted_class,
            "confidence": confidence_score,
            "treatment": treatment
        }

    return render_template(
        "index.html",
        result=result,
        uploaded_image=uploaded_image
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)