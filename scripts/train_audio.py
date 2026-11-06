"""Train a small audio CNN (drone / helicopter / background) and measure its confusion matrix.

    python scripts/train_audio.py --audio path/to/Drone-detection-dataset/Data/Audio

Evaluation is k-fold *by file*: every 10-second clip is tested by a model that never heard any
part of it, so the matrix is honest even though the dataset has only 90 clips. The result
replaces the drone, helicopter and background rows of configs/confusion/acoustic.json; the bird
and aircraft rows stay as documented assumptions (the dataset has no such audio).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing
import numpy as np

from nabhnetra.common import ROOT
from nabhnetra.realdata import AUDIO_LABELS, kfold_by_group, load_wav, logmel, parse_audio_name, windows


def features(files, sr, win_s, hop_s):
    X, y, g = [], [], []
    for f in files:
        label, _ = parse_audio_name(f.stem)
        wav, _ = load_wav(f, sr)
        for w in windows(wav, sr, win_s, hop_s):
            m = logmel(w, sr)
            X.append((m - m.mean()) / (m.std() + 1e-6))
            y.append(AUDIO_LABELS.index(label))
            g.append(f.stem)
    return np.stack(X)[:, None], np.array(y), np.array(g)


def make_model(n_mels, n_classes):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.MaxPool2d(2))

    return nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))


def train(X, y, epochs, seed, device):
    import torch

    torch.manual_seed(seed)
    model = make_model(X.shape[2], len(AUDIO_LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for idx in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // 32)):
            xb = Xt[idx].clone()
            # SpecAugment-style masking: hide a random band of frequencies and of time
            f0 = int(rng.integers(0, xb.shape[2] - 8))
            t0 = int(rng.integers(0, max(1, xb.shape[3] - 10)))
            xb[:, :, f0:f0 + 8, :] = 0
            xb[:, :, :, t0:t0 + 10] = 0
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yt[idx].to(device))
            loss.backward()
            opt.step()
    return model


def predict(model, X, device):
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--audio", required=True, help="the dataset's Data/Audio folder")
    ap.add_argument("--out", default="configs/confusion/acoustic.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--win", type=float, default=1.0, help="window length in seconds")
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", help="also train on all files and save the weights here")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(p for p in Path(args.audio).glob("*.wav") if parse_audio_name(p.stem))
    if not files:
        raise SystemExit(f"No audio files found in {args.audio}")
    print(f"{len(files)} files, extracting log-mel windows ...")
    X, y, groups = features(files, args.sr, args.win, args.hop)
    labels_of = {f.stem: parse_audio_name(f.stem)[0] for f in files}
    fold_of = kfold_by_group(list(labels_of), list(labels_of.values()), args.folds, args.seed)
    folds = np.array([fold_of[g] for g in groups])

    counts = np.zeros((len(AUDIO_LABELS), len(AUDIO_LABELS)), dtype=int)
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        model = train(X[tr], y[tr], args.epochs, args.seed + k, device)
        pred = predict(model, X[te], device)
        for t, p in zip(y[te], pred):
            counts[t, p] += 1
        print(f"  fold {k + 1}/{args.folds}: window accuracy {(pred == y[te]).mean():.3f}")

    measured = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)
    print("\nRows true, columns predicted:", AUDIO_LABELS)
    print(counts)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    rows = dict(zip(data["true_classes"], data["matrices"]["all"]))
    rows["drone"] = measured[AUDIO_LABELS.index("drone")].round(4).tolist()
    rows["helicopter"] = measured[AUDIO_LABELS.index("helicopter")].round(4).tolist()
    data.update({
        "source": (f"measured {date.today().isoformat()}: {args.folds}-fold by file over "
                   f"{len(files)} clips ({len(y)} windows of {args.win} s). Drone, helicopter and "
                   "background rows measured; bird and aircraft rows assumed."),
        "labels": list(AUDIO_LABELS),
        "true_classes": list(rows),
        "matrices": {"all": list(rows.values())},
        "background_row": measured[AUDIO_LABELS.index("background")].round(4).tolist(),
        "assumed_rows": ["bird", "aircraft"],
        "counts_true_x_pred": counts.tolist(),
    })
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.save_model:
        model = train(X, y, args.epochs, args.seed, device)
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved {args.save_model}")


if __name__ == "__main__":
    main()
