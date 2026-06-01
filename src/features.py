import numpy as np
import logging as log
import librosa
from typing import Tuple, List


        
def extract_features(processed_sample: List[Tuple[np.ndarray, int, int]],
                     sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> List[Tuple[np.ndarray, int, int]]:
    """Extracts MFCC features from a preprocessed sample, then computes delta and delta-delta factors

    Args:
        processed_sample: list of tuples (segment, label) from preprocessing
        sr: sample rate
        n_mfcc: number of mfcc coefficients
        n_fft: FFT window size
        hop_length: a step between windows

    Returns:
        List of tuples (feature_vector, label) where feature_vector is np.ndarray
        shape (n_mfcc * 6,) concatenation of mean and std for each MFCC, delta nad delta-delta coefficient,
        label is int number - 0 or 1
    """

    try:  
        feature_vector = []
        for sample in processed_sample:
            mfcc_features = librosa.feature.mfcc(y = sample[0], 
                            sr = sr, n_mfcc = n_mfcc,
                            n_fft = n_fft, hop_length = hop_length)
            delta = librosa.feature.delta(mfcc_features)
            delta2 = librosa.feature.delta(mfcc_features, order=2)
            
            mean = np.mean(mfcc_features, axis = 1)
            std = np.std(mfcc_features, axis = 1)
            mean_d = np.mean(delta, axis = 1)
            std_d = np.std(delta, axis = 1)
            mean_d2 = np.mean(delta2, axis = 1)
            std_d2 = np.std(delta2, axis = 1)
            
            feature = np.concatenate([mean, std])
            feature_d = np.concatenate([mean_d, std_d])
            feature_d2 = np.concatenate([mean_d2, std_d2])
            
            full_feature = np.concatenate([feature, feature_d, feature_d2])
            feature_vector.append((full_feature, sample[1], sample[2]))

        return feature_vector
    
    except Exception as e:
        log.error(f"Error extracting features: {e}")
        return []
    
    
