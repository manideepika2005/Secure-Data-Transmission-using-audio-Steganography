import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/audio_steganalysis_cnn.h5"
N_MFCC = 40
SAMPLE_RATE = 22050
DURATION = 2

model = tf.keras.models.load_model(MODEL_PATH)

def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)
    return "STEGO AUDIO" if np.argmax(prediction) == 1 else "COVER AUDIO"
