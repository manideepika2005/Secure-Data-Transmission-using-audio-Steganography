import os
import sys
from collections import Counter

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"
SAMPLE_RATE = 22050
DURATION = 2
N_MFCC = 40


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)


X, y = [], []

for label in ["cover", "stego"]:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        print(f"Error: expected folder '{folder}' not found.")
        sys.exit(1)
    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            X.append(extract_features(os.path.join(folder, file)))
            y.append(label)

if len(X) == 0:
    print(f"Error: no .wav files found under '{DATASET_PATH}'.")
    sys.exit(1)

counts = Counter(y)
print("Dataset counts:")
for k, v in counts.items():
    print(f" - {k}: {v}")

# Require at least one sample per class and at least 2 total samples
if any(v < 1 for v in counts.values()):
    print("Error: each class must contain at least one sample.")
    sys.exit(1)

X = np.array(X)

# Encode labels and stratify split to keep class distribution
le = LabelEncoder()
y_int = le.fit_transform(y)

try:
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int, test_size=0.2, stratify=y_int, random_state=42
    )
except ValueError:
    # Fall back to non-stratified split if stratify not possible
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int, test_size=0.2, random_state=42
    )

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

num_classes = len(le.classes_)
y_train = to_categorical(y_train_int, num_classes=num_classes)
y_test = to_categorical(y_test_int, num_classes=num_classes)

print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# Build model with explicit Input to avoid input_shape warning
model = tf.keras.Sequential([
    tf.keras.Input(shape=(N_MFCC, 1)),
    tf.keras.layers.Conv1D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(le.classes_), activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Ensure models directory exists and save in native Keras format (with HDF5 for compatibility)
os.makedirs("models", exist_ok=True)
keras_path = "models/audio_steganalysis_cnn.keras"
h5_path = "models/audio_steganalysis_cnn.h5"
model.save(keras_path)
try:
    model.save(h5_path)
except Exception:
    # Some TF versions may warn; ignore if HDF5 saving isn't available
    pass
print("✅ Model Saved")
