from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    image_name = db.Column(db.String(200))
    disease = db.Column(db.String(200))
    confidence = db.Column(db.String(50))
    remedy = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )