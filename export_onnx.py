"""Export the whole pipeline (waveform -> log-mel -> both CNNs) to one ONNX file,
so the web page in docs/ can run it in the browser with onnxruntime-web.

The STFT is written as a 1-D convolution with a fixed DFT basis so that it exports cleanly
and matches torchaudio.transforms.MelSpectrogram exactly (checked below).

Usage:  python export_onnx.py      (after train.py)
"""
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from features import EMOTIONS, SR, N_FFT, HOP, N_MELS, N_SAMPLES, log_mel
from models import ScratchCNN, ResNetEmbedder, transfer_head


class LogMelFrontend(nn.Module):
    """Same result as features.log_mel, but for a batch of waveforms and ONNX-friendly."""

    def __init__(self):
        super().__init__()
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
        window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
        angle = 2 * np.pi * k[:, None] * n[None, :] / N_FFT
        basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * window   # (2*201, 400)
        self.register_buffer("basis", basis.float().unsqueeze(1))
        fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, SR / 2, N_MELS, SR)
        self.register_buffer("fb", fb.T.contiguous())                       # (64, 201)

    def forward(self, wav):                                   # wav: (B, 48000)
        x = F.pad(wav.unsqueeze(1), (N_FFT // 2, N_FFT // 2), mode="reflect")
        spec = F.conv1d(x, self.basis, stride=HOP)             # (B, 402, 301)
        re, im = spec[:, :N_FFT // 2 + 1], spec[:, N_FFT // 2 + 1:]
        mel = torch.matmul(self.fb, re * re + im * im)         # (B, 64, 301)
        db = 10 * torch.log10(torch.clamp(mel, min=1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80)
        mean = db.mean(dim=(1, 2), keepdim=True)
        std = ((db - mean) ** 2).sum(dim=(1, 2), keepdim=True) / (db.shape[1] * db.shape[2] - 1)
        return ((db - mean) / (torch.sqrt(std) + 1e-6)).unsqueeze(1)   # (B, 1, 64, 301)


class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = LogMelFrontend()
        self.scratch = ScratchCNN(len(EMOTIONS))
        self.scratch.load_state_dict(torch.load("weights/scratch_cnn.pt", map_location="cpu"))
        self.embedder = ResNetEmbedder()
        self.head = transfer_head(len(EMOTIONS))
        self.head.load_state_dict(torch.load("weights/transfer_head.pt", map_location="cpu"))
        self.eval()

    def forward(self, wav):
        x = self.frontend(wav)
        return (x[:, 0],
                torch.softmax(self.scratch(x), 1),
                torch.softmax(self.head(self.embedder.net(self._to_rgb(x))), 1))

    @staticmethod
    def _to_rgb(x):
        from models import IMAGENET_MEAN, IMAGENET_STD
        lo = x.amin(dim=(2, 3), keepdim=True)
        hi = x.amax(dim=(2, 3), keepdim=True)
        x = ((x - lo) / (hi - lo + 1e-6)).repeat(1, 3, 1, 1)
        return (x - IMAGENET_MEAN) / IMAGENET_STD


def main():
    model = Pipeline()
    wav = torch.randn(1, N_SAMPLES) * 0.1
    torch.onnx.export(model, (wav,), "docs/emotion.onnx", input_names=["waveform"],
                      output_names=["spectrogram", "p_scratch", "p_transfer"],
                      opset_version=17, dynamo=False)

    # Check: the ONNX pipeline must agree with the PyTorch code used for training and evaluation.
    from features import load_wav, trim_silence, fix_length
    y = fix_length(trim_silence(load_wav("data/emodb/wav/03a01Wa.wav")))
    sess = ort.InferenceSession("docs/emotion.onnx")
    spec, p1, p2 = sess.run(None, {"waveform": y[None].astype(np.float32)})
    with torch.no_grad():
        ref = log_mel(y).unsqueeze(0)
        r1 = torch.softmax(model.scratch(ref), 1).numpy()
        r2 = torch.softmax(model.head(model.embedder(ref)), 1).numpy()
    print("max |spec diff|", np.abs(spec - ref[:, 0].numpy()).max())
    print("max |p diff| scratch", np.abs(p1 - r1).max(), "transfer", np.abs(p2 - r2).max())
    print("prediction:", EMOTIONS[p1.argmax()], EMOTIONS[p2.argmax()])


if __name__ == "__main__":
    main()
