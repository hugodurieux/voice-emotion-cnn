"""Web app: record your voice, see its waveform and spectrogram, get a prediction from both CNNs.
Run locally with `python app.py`; the same file runs as a Hugging Face Space."""
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from features import EMOTIONS, SR, HOP, to_mono_16k, trim_silence, fix_length, log_mel
from models import ScratchCNN, ResNetEmbedder, transfer_head

scratch = ScratchCNN(len(EMOTIONS))
scratch.load_state_dict(torch.load("weights/scratch_cnn.pt", map_location="cpu"))
scratch.eval()
embedder = ResNetEmbedder()
head = transfer_head(len(EMOTIONS))
head.load_state_dict(torch.load("weights/transfer_head.pt", map_location="cpu"))
head.eval()


def plot_signal(y, spec):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [1, 2]})
    t = np.arange(len(y)) / SR
    ax1.plot(t, y, lw=0.5, color="#2a6f97")
    ax1.set_xlim(0, t[-1]); ax1.set_ylabel("amplitude"); ax1.set_title("Waveform (3 s window, 16 kHz)")
    ax2.imshow(spec, origin="lower", aspect="auto", cmap="magma",
               extent=[0, spec.shape[1] * HOP / SR, 0, spec.shape[0]])
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("mel band"); ax2.set_title("Log-mel spectrogram (CNN input, 64 x 301)")
    fig.tight_layout()
    return fig


@torch.no_grad()
def analyse(audio):
    if audio is None:
        raise gr.Error("Record or upload some audio first.")
    sr, y = audio
    y = to_mono_16k(y, sr)
    if np.abs(y).max() < 1e-3:
        raise gr.Error("The recording is silent - check your microphone.")
    y = y / np.abs(y).max() * 0.9          # loudness normalisation
    y = fix_length(trim_silence(y))
    x = log_mel(y).unsqueeze(0)            # (1, 1, 64, 301)
    p1 = torch.softmax(scratch(x), 1)[0].numpy()
    p2 = torch.softmax(head(embedder(x)), 1)[0].numpy()
    fig = plot_signal(y, x[0, 0].numpy())
    return (fig,
            {e: float(p) for e, p in zip(EMOTIONS, p1)},
            {e: float(p) for e, p in zip(EMOTIONS, p2)})


with gr.Blocks(title="Voice Emotion CNN") as demo:
    gr.Markdown(
        "# Can a machine hear how you feel?\n"
        "Record a sentence **with feeling** (or upload a WAV), then press **Analyse**. "
        "Two CNNs trained on the German EmoDB corpus (7 emotions) and evaluated on speakers they never heard "
        "give their answer. Try acting anger, sadness, happiness, fear, disgust, boredom or neutral.")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your voice")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label="Signal")
    with gr.Row():
        out1 = gr.Label(num_top_classes=7, label="Model 1 - CNN from scratch")
        out2 = gr.Label(num_top_classes=7, label="Model 2 - ResNet-18 transfer learning")
    btn.click(analyse, inputs=audio, outputs=[plot, out1, out2])
    audio.stop_recording(analyse, inputs=audio, outputs=[plot, out1, out2])
    gr.Markdown("The models were trained on acted German speech from 10 speakers, so expect mistakes on "
                "other languages, microphones and natural speech. Audio is processed in memory and not stored.")

if __name__ == "__main__":
    demo.launch()
