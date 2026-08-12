import logging as log
from pathlib import Path

import hydra
import numpy as np
from datasets import Audio, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from evaluation import evaluation
from features import extract_features
from preprocess import AudioPreprocessing, build_ambient_pool
from rf import train


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    log.basicConfig(level=log.INFO)

    ds = load_dataset("geronimobasso/drone-audio-detection-samples")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

    ambient_pool = build_ambient_pool(ds, seed=cfg.seed)
    preprocessor = AudioPreprocessing(ambient_pool=ambient_pool)

    root = Path(__file__).resolve().parents[1]
    if (root / "data" / "features.npz").exists():
        with np.load(root / "data" / "features.npz") as features:
            X = features["X"]
            y = features["y"]
            groups = features["groups"]
    else:
        features = []
        for i in tqdm(range(len(ds["train"]))):
            sample = ds["train"][i]["audio"]
            label = ds["train"][i]["label"]

            preprocessed_sample = preprocessor(sample, label, group_id=i)
            feature_vector = extract_features(
                preprocessed_sample,
                cfg.data.sr,
                cfg.data.n_mfcc,
                cfg.data.n_fft,
                cfg.data.hop_length,
            )
            features.append(feature_vector)

        X = np.array([item[0] for sublist in features for item in sublist])
        y = np.array([item[1] for sublist in features for item in sublist])
        groups = np.array([item[2] for sublist in features for item in sublist])

        np.savez(root / "data" / "features.npz", X=X, y=y, groups=groups)

    classifier, _, y_pred, X_test, y_test, X_val, y_val = train(
        X, y, groups, cfg.model.random_state, cfg.model.n_estimators
    )

    results = evaluation(X_test, y_test, y_pred, classifier, X_val, y_val)


if __name__ == "__main__":
    main()
