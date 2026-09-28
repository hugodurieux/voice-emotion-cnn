"""Train and evaluate both CNNs on EmoDB with leave-speakers-out cross-validation,
then retrain on all speakers and save the weights used by the web app.

Usage:  python train.py            (expects data/emodb/wav/*.wav)
"""
import json
import os
import random
import time
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from features import EMOTIONS, EMODB_CODES, load_wav, trim_silence, fix_length, log_mel, N_SAMPLES
from models import ScratchCNN, ResNetEmbedder, transfer_head

torch.set_num_threads(max(1, os.cpu_count() - 1))
SEED = 0
EPOCHS_SCRATCH = 40
EPOCHS_HEAD = 200
BATCH = 32

# Leave-2-speakers-out: each fold holds out one male and one female speaker.
FOLDS = [("03", "08"), ("10", "09"), ("11", "13"), ("12", "14"), ("15", "16")]


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_emodb(root="data/emodb/wav"):
    waves, labels, speakers = [], [], []
    for path in sorted(glob(os.path.join(root, "*.wav"))):
        name = os.path.basename(path)
        waves.append(trim_silence(load_wav(path)))
        labels.append(EMOTIONS.index(EMODB_CODES[name[5]]))
        speakers.append(name[:2])
    return waves, np.array(labels), np.array(speakers)


def spec_augment(x):
    """Mask one random frequency stripe and one random time stripe per example."""
    x = x.clone()
    for i in range(x.shape[0]):
        f = random.randint(0, 8); f0 = random.randint(0, x.shape[2] - f)
        t = random.randint(0, 30); t0 = random.randint(0, x.shape[3] - t)
        x[i, :, f0:f0 + f, :] = 0
        x[i, :, :, t0:t0 + t] = 0
    return x


def batch_specs(waves, idx, augment):
    specs = []
    for i in idx:
        y = waves[i]
        if augment:   # random time shift: place the clip at a random position in the 3 s window
            room = abs(len(y) - N_SAMPLES)
            specs.append(log_mel(fix_length(y, offset=random.randint(0, room))))
        else:
            specs.append(log_mel(fix_length(y)))
    x = torch.stack(specs)
    return spec_augment(x) if augment else x


def class_weights(y):
    counts = np.bincount(y, minlength=len(EMOTIONS)).astype(np.float32)
    return torch.tensor(counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1)))


def train_scratch(waves, labels, train_idx):
    seed_all()
    model = ScratchCNN(len(EMOTIONS))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=EPOCHS_SCRATCH * ((len(train_idx) + BATCH - 1) // BATCH))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels[train_idx]))
    for epoch in range(EPOCHS_SCRATCH):
        model.train()
        perm = np.random.permutation(train_idx)
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            x = batch_specs(waves, idx, augment=True)
            y = torch.from_numpy(labels[idx])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict_scratch(model, waves, idx):
    return model(batch_specs(waves, idx, augment=False)).argmax(1).numpy()


def train_head(emb, labels, train_idx):
    seed_all()
    head = transfer_head(len(EMOTIONS))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels[train_idx]))
    x, y = emb[train_idx], torch.from_numpy(labels[train_idx])
    for _ in range(EPOCHS_HEAD):
        head.train()
        perm = torch.randperm(len(x))
        for b in range(0, len(x), BATCH):
            i = perm[b:b + BATCH]
            opt.zero_grad()
            loss_fn(head(x[i]), y[i]).backward()
            opt.step()
    return head.eval()


def plot_confusion(cm, title, path):
    cmn = 100 * cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{cmn[i, j]:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 50 else "black", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_results(res, path):
    names = ["Chance", "CNN from scratch", "ResNet-18 transfer"]
    uars = [100 / len(EMOTIONS), 100 * res["scratch"]["uar_mean"], 100 * res["transfer"]["uar_mean"]]
    stds = [0, 100 * res["scratch"]["uar_std"], 100 * res["transfer"]["uar_std"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, uars, yerr=stds, capsize=6, color=["#bbbbbb", "#2a6f97", "#e07a5f"])
    for bar, v in zip(bars, uars):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f} %", ha="center")
    ax.set_ylabel("Unweighted average recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title("EmoDB, 7 emotions, unseen speakers (5-fold)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True); os.makedirs("weights", exist_ok=True)
    t0 = time.time()
    waves, labels, speakers = load_emodb()
    print(f"{len(waves)} clips, {len(set(speakers))} speakers, classes: {np.bincount(labels)}")

    print("Computing frozen ResNet-18 embeddings...")
    embedder = ResNetEmbedder()
    emb = torch.cat([embedder(batch_specs(waves, range(i, min(i + BATCH, len(waves))), False))
                     for i in range(0, len(waves), BATCH)])

    res = {"scratch": {"fold_uar": [], "fold_acc": []}, "transfer": {"fold_uar": [], "fold_acc": []}}
    preds = {"scratch": np.zeros_like(labels), "transfer": np.zeros_like(labels)}
    for k, held_out in enumerate(FOLDS):
        test = np.where(np.isin(speakers, held_out))[0]
        train = np.where(~np.isin(speakers, held_out))[0]
        assert not set(speakers[test]) & set(speakers[train])   # no speaker in two splits

        preds["scratch"][test] = predict_scratch(train_scratch(waves, labels, train), waves, test)
        head = train_head(emb, labels, train)
        with torch.no_grad():
            preds["transfer"][test] = head(emb[test]).argmax(1).numpy()

        for m in preds:
            res[m]["fold_uar"].append(balanced_accuracy_score(labels[test], preds[m][test]))
            res[m]["fold_acc"].append(accuracy_score(labels[test], preds[m][test]))
        print(f"fold {k + 1} (test speakers {held_out}): "
              f"scratch UAR {res['scratch']['fold_uar'][-1]:.3f} | "
              f"transfer UAR {res['transfer']['fold_uar'][-1]:.3f}  [{time.time() - t0:.0f}s]")

    for m, title in [("scratch", "CNN from scratch"), ("transfer", "ResNet-18 transfer")]:
        res[m]["uar_mean"] = float(np.mean(res[m]["fold_uar"]))
        res[m]["uar_std"] = float(np.std(res[m]["fold_uar"]))
        res[m]["acc_mean"] = float(np.mean(res[m]["fold_acc"]))
        cm = confusion_matrix(labels, preds[m], labels=range(len(EMOTIONS)))
        res[m]["confusion"] = cm.tolist()
        plot_confusion(cm, f"{title} - EmoDB, unseen speakers (row %)", f"results/confusion_{m}.png")
        print(f"{title}: UAR {res[m]['uar_mean']:.3f} +/- {res[m]['uar_std']:.3f}, "
              f"accuracy {res[m]['acc_mean']:.3f}")
    res["folds"] = FOLDS
    res["emotions"] = EMOTIONS
    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=2)
    plot_results(res, "results/uar_comparison.png")

    print("Retraining on all 10 speakers for the app...")
    everyone = np.arange(len(waves))
    torch.save(train_scratch(waves, labels, everyone).state_dict(), "weights/scratch_cnn.pt")
    torch.save(train_head(emb, labels, everyone).state_dict(), "weights/transfer_head.pt")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
