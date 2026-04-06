"""
Audio processing for training: mel spectrogram extraction.
Uses the reference-compatible mel frontend parameters.
"""

import os
import tempfile

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "lipsync_numba_cache"))
import librosa
from scipy import signal


class AudioProcessor:
    def __init__(self, cfg):
        self.sample_rate = cfg["sample_rate"]
        self.n_fft = cfg["n_fft"]
        self.hop_size = cfg["hop_size"]
        self.win_size = cfg["win_size"]
        self.n_mels = cfg["n_mels"]
        self.fmin = cfg["fmin"]
        self.fmax = cfg["fmax"]
        self.preemphasis_k = cfg["preemphasis"]

        self._mel_basis = librosa.filters.mel(
            sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
        )

    def load_wav(self, path):
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav

    def melspectrogram(self, wav):
        """Compute normalized mel spectrogram matching the reference setup."""
        wav = signal.lfilter([1, -self.preemphasis_k], [1], wav)
        D = librosa.stft(y=wav, n_fft=self.n_fft, hop_length=self.hop_size,
                         win_length=self.win_size)
        S = np.dot(self._mel_basis, np.abs(D))

        # Amp to dB
        min_level = np.exp(-100 / 20 * np.log(10))
        S = 20 * np.log10(np.maximum(min_level, S)) - 20  # ref_level_db=20

        # Symmetric normalize to [-4, 4]
        S = np.clip((2 * 4.0) * ((S - (-100)) / 100.0) - 4.0, -4.0, 4.0)
        return S.astype(np.float32)

    def mel_chunks(self, mel, fps, mel_step_size=16):
        """Slice mel into per-frame chunks."""
        mel_idx_mult = (self.sample_rate / float(self.hop_size)) / float(fps)
        chunks = []
        i = 0
        while True:
            start = int(i * mel_idx_mult)
            if start + mel_step_size > mel.shape[1]:
                chunks.append(mel[:, -mel_step_size:])
                break
            chunks.append(mel[:, start:start + mel_step_size])
            i += 1
        return chunks
