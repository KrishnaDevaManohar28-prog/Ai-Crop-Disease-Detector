import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import json
from PIL import Image
import os

dataset_path = r"data set\PlantVillage"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Deleting corrupted file:", path)
            os.remove(path)
DATASET_PATH = r"data set\PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 38

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", subset="training")
val_gen = train_datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", subset="validation")

labels = {v: k for k, v in train_gen.class_indices.items()}
json.dump(labels, open("model/class_labels.json","w"), indent=2)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
out = Dense(16, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

cb = [ModelCheckpoint("model/model.h5", monitor="val_accuracy", save_best_only=True),
      EarlyStopping(monitor="val_loss", patience=3)]
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=cb)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy"); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss"); plt.legend()
plt.savefig("model/training_graphs.png")

loss, acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {acc*100:.2f}%")
