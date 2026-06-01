import sounddevice as sd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from features import extract_features
from omegaconf import OmegaConf



root = Path(__file__).resolve().parents[1]
model = joblib.load(root / 'models' / "rf.joblib")
scaler = joblib.load(root / 'models' / "rf_scaler.joblib")
buffer = np.zeros(16000)

cfg = OmegaConf.load(root / 'configs/data/mfcc.yaml')

def callback(indata, frames, time, status):
    if status:
        print(status)
    
    audio_chunk = indata[:, 0] 

    global buffer
    buffer = np.roll(buffer, -4000)
    buffer[-4000:] = audio_chunk

    sample = [(buffer, 0, 0)]
    feature_vector = extract_features(sample, cfg.sr, 
                                      cfg.n_mfcc, cfg.n_fft, 
                                      cfg.hop_length)[0][0]

    X_new = scaler.transform(feature_vector.reshape(1, -1))
    y_pred = model.predict(X_new)

    print(f"Amplitude: {np.max(np.abs(audio_chunk)):.4f} | Prediction: {y_pred[0]}")
    
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    blocksize=4000,
    device=6,
    callback=callback
)


with stream:
    sd.sleep(1000000)



