from tensorflow.keras.models import load_model

model = load_model("model/model.h5", compile=False)

model.save("model/new_model.h5")