from pymongo import MongoClient
from datetime import datetime

client     = MongoClient("mongodb://localhost:27017/")
db         = client["crop_disease_db"]
collection = db["predictions"]

def save_prediction(data: dict):
    try:
        collection.insert_one(data)
    except Exception as e:
        print(f"DB Error: {e}")

def get_all_predictions():
    return list(collection.find({}, {"_id": 0}))
