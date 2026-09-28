"""Audio -> log-mel spectrogram, shared by training and the web app."""
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly
from math import gcd

SR = 16000            # 16 kHz mono
DURATION = 3.0        # 3-second window
N_SAMPLES = int(SR * DURATION)
N_FFT = 400           # 25 ms frames
HOP = 160             # every 10 ms
N_MELS = 64           # 64 mel bands -> spectrogram of 64 x 301

EMOTIONS = ["anger", "boredom", "disgust", "fear", "happiness", "sadness", "neutral"]
# EmoDB file name letter (German) -> emotion
EMODB_CODES = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
               "F": "happiness", "T": "sadness", "N": "neutral"}

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP, n_mels=N_MELS)
_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)


def to_mono_16k(y, sr):
    """Convert any waveform (int or float, mono or stereo) to float32 mono at 16 kHz."""
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y / np.iinfo(y.dtype).max
    y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return y


def load_wav(path):
    y, sr = sf.read(path)
    return to_mono_16k(y, sr)


def fix_length(y, offset=None):
    """Crop or zero-pad to exactly 3 s. Centered unless an offset is given."""
    if len(y) >= N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2 if offset is None else min(offset, len(y) - N_SAMPLES)
        return y[start:start + N_SAMPLES]
    out = np.zeros(N_SAMPLES, dtype=np.float32)
    start = (N_SAMPLES - len(y)) // 2 if offset is None else min(offset, N_SAMPLES - len(y))
    out[start:start + len(y)] = y
    return out


def trim_silence(y, thresh_db=-40.0):
    """Drop leading/trailing silence (useful for browser recordings)."""
    if len(y) == 0:
        return y
    frame = 400
    n = len(y) // frame
    if n == 0:
        return y
    energy = (y[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-10
    db = 10 * np.log10(energy / energy.max())
    keep = np.where(db > thresh_db)[0]
    if len(keep) == 0:
        return y
    return y[keep[0] * frame:(keep[-1] + 1) * frame]


def log_mel(y):
    """3-second waveform -> standardised log-mel tensor of shape (1, 64, 301)."""
    m = _to_db(_mel(torch.from_numpy(y)))
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m.unsqueeze(0)
