import logging as log
from typing import Any

import librosa
import numpy as np
import scipy.signal

logger = log.getLogger(__name__)


def load_audio(sample: Any) -> tuple[np.ndarray, int]:
    """
    Loads audio from a HuggingFace dataset sample

    Args:
        sample: AudioDecoder object from torchcodec

    Returns:
        Tuple of (audio array, sample rate)
    """

    a_data = sample.get_all_samples()
    array = a_data.data.numpy().squeeze()
    sr = a_data.sample_rate

    if a_data.sample_rate != 16000:
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000

    return (array, sr)


def build_ambient_pool(ds: Any, seed: int, n: int = 500) -> list[np.ndarray]:
    """
    Builds ambient pool from random no-drone audio files

    Args:
        ds: HuggingFace dataset object
        seed: int number for random generation
        n: number of ambients

    Returns:
        List of [np.ndarray] no-drone audio sounds
    """

    no_drone_label = np.array(ds["train"]["label"])
    no_drone_indices = np.where(no_drone_label == 0)[0]
    no_drone_sample = ds["train"].select(no_drone_indices)
    no_drone_pool = no_drone_sample.shuffle(seed=seed).select(range(n))

    mapped = no_drone_pool.map(
        lambda x: {"array": load_audio(x["audio"])[0]}, num_proc=1
    )
    ambient_pool = [np.array(a, dtype=np.float32) for a in mapped["array"]]

    return ambient_pool


class AudioPreprocessing:
    """
    Class for preprocessing audio to prepare it for creating features.
    """

    def __init__(
        self,
        ambient_pool=None,
        min_duration_s=1.0,
        p=0.5,
        window_size_s=1.0,
        stride_s=0.25,
        low=80,
        high=6000,
    ):
        """
        Includes padding, augmenting pre and post cut and segmenting

        Args:
            ambient_pool:
            min_duration_s: minimum duration of a sound (seconds)
            p: augmentation probability
            window_size_s: window size in seconds for segmentation
            stride_s: stride in seconds for segmentation
            low: lowest fake background noise frequency
            high: highest background noise frequency
        """
        if ambient_pool is None:
            raise ValueError("ambient_pool must not be None")

        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1")

        if low >= high:
            raise ValueError("low must have a lower value than high")

        if stride_s >= window_size_s:
            raise ValueError("stride_s must have lower value than window_size_s")

        self.ambient_pool = ambient_pool
        self.min_duration_s = min_duration_s
        self.p = p
        self.window_size_s = window_size_s
        self.stride_s = stride_s
        self.low = low
        self.high = high

    def __call__(
        self, sample: Any, label: int, group_id: int
    ) -> list[tuple[np.ndarray, int, int]]:
        """
        End-to-end pipeline for preprocessing of audio.
        Loads, pads, augments and segments audio.

        Args:
            sample: AudioDecoder object from torchcodec
            label: 0 or 1 label from dataset

        Returns:
            List of [np.ndarray] or [None] if pipeline fails
        """

        try:
            array, sr = load_audio(sample)

            array = self.pad_audio(array, sr)
            array = self.augment_pre_cut(array, sr)
            segments = self.segment_audio(array, sr)

            processed_segments = []

            for seg in segments:
                seg = self.augment_post_cut(seg, sr)
                sl = (seg, label, group_id)
                processed_segments.append(sl)

            return processed_segments

        except Exception as e: # noqa: BLE001
            logger.error(f"Error processing sample: {e}")
            return []

    def pad_audio(
        self, array: np.ndarray, sr: int, min_duration_s: float | None = None
    ) -> np.ndarray:
        """
        Stretches the array with ambient sounds

        Args:
            array: np.ndarray audio
            sr: sample rate
            min_duration_s: optional duration parameter, default = None

        Returns:
            Padded np.ndarray audio
        """

        min_dur = min_duration_s if min_duration_s is not None else self.min_duration_s
        if (len(array) / sr) >= min_dur:
            return array

        padded_array = array.copy()

        ambient = self.ambient_pool[np.random.randint(len(self.ambient_pool))]

        min_samples = int(min_dur * sr)
        remaining = min_samples - len(padded_array)
        repeats = int(np.ceil(remaining / len(ambient)))
        padded_array = np.concatenate([padded_array, np.tile(ambient, repeats)])

        return padded_array[:min_samples]

    def augment_pre_cut(self, array: np.ndarray, sr: int) -> np.ndarray:
        """
        Applies pre-cut audio augmentation (ambient mix, noise, reverb)

        Args:
            array: np.ndarray audio
            sr: sample rate

        Returns:
            Augmented and normalized np.ndarray audio
        """

        ambient = self.ambient_pool[np.random.randint(len(self.ambient_pool))]

        if len(ambient) > len(array):
            ambient = ambient[: len(array)]

        if len(ambient) < len(array):
            ambient = self.pad_audio(ambient, sr, min_duration_s=len(array) / sr)

        gain = np.random.uniform(0.1, 0.5)
        amplitude = np.random.uniform(0.005, 0.05)

        output = array.copy()

        if np.random.random() < self.p:
            output = output + ambient * gain

        if np.random.random() < self.p:
            output = (
                output + np.random.randn(len(output)).astype(np.float32) * amplitude
            )

        if np.random.random() < self.p:
            ir = np.random.randn(int(sr * 0.5)) * np.exp(
                -np.linspace(0, 5, int(sr * 0.5))
            )
            output = scipy.signal.fftconvolve(output, ir, mode="same").astype(
                np.float32
            )

        if np.max(np.abs(output)) == 0:
            logger.warning("Silent segment detected")
            return output

        final_pre_cut = output / np.max(np.abs(output))

        return final_pre_cut

    def segment_audio(self, array: np.ndarray, sr: int) -> list[np.ndarray]:
        """
        Splits audio into overlapping segments with padding

        Args:
            array: np.ndarray audio
            sr: sample rate

        Returns:
            List of np.ndarray audio segments
        """

        window_size = int(self.window_size_s * sr)
        stride = int(self.stride_s * sr)

        segments = []

        for i in range(0, len(array), stride):
            segment = array[i : i + window_size]
            if len(segment) < window_size:
                segment = self.pad_audio(segment, sr, min_duration_s=self.window_size_s)
            segments.append(segment)

        return segments

    def augment_post_cut(self, array: np.ndarray, sr: int) -> np.ndarray:
        """
        Applies post-cut audio augmentation (gain, shift, 
        filter, stretch, pitch)

        Args:
            array: np.ndarray audio
            sr: sample rate

        Returns:
            Augmented and normalized np.ndarray audio
        """
        output = array.copy()

        if np.random.random() < self.p:
            output = output * np.random.uniform(0.5, 1.5)

        if np.random.random() < self.p:
            shift_samples = int(np.random.uniform(-0.1, 0.1) * sr)
            if shift_samples < 0:
                output = np.pad(output, (abs(shift_samples), 0), mode="constant")
            else:
                output = np.pad(output, (0, shift_samples), mode="constant")
            output = output[: len(array)]

        if np.random.random() < self.p:
            sos = scipy.signal.butter(
                N=4, Wn=[self.low, self.high], btype="band", fs=sr, output="sos"
            )
            output = scipy.signal.sosfilt(sos, output).astype(np.float32)

        if np.random.random() < self.p:
            output = librosa.effects.time_stretch(
                output, rate=np.random.uniform(0.8, 1.2)
            )

        if np.random.random() < self.p:
            output = librosa.effects.pitch_shift(
                output, sr=sr, n_steps=np.random.randint(-3, 4)
            )

        if len(output) > (self.window_size_s * sr):
            output = output[: int(self.window_size_s * sr)]

        if len(output) < (self.window_size_s * sr):
            output = self.pad_audio(output, sr, min_duration_s=self.window_size_s)

        if np.max(np.abs(output)) == 0:
            logger.warning("Silent segment detected")
            return output

        final_post_cut = output / np.max(np.abs(output))

        return final_post_cut
