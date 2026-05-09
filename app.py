from flask import Flask, render_template, request
from db import db, Prediction

from datetime import datetime
import os

app = Flask(__name__)

# =========================
# DATABASE CONFIG
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
        # DEMO AI PREDICTION
        # =========================
        predicted_class = "Tomato Early Blight"

        confidence_score = 98

        treatment = """
        Use fungicide spray.
        Remove affected leaves.
        Avoid overwatering.
        """

        # =========================
        # SAVE TO DATABASE
        # =========================
        new_prediction = Prediction(
            image_name=filename,
            disease=predicted_class,
            confidence=str(confidence_score),
            remedy=treatment,
            created_at=datetime.now()
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
    app.run(host="0.0.0.0", port=5000)