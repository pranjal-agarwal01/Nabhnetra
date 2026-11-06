"""Train a small audio CNN (drone / helicopter / background) and measure its confusion matrix.

    python scripts/train_audio.py --audio path/to/Drone-detection-dataset/Data/Audio

Evaluation is k-fold *by file*: every 10-second clip is tested by a model that never heard any
part of it, so the matrix is honest even though the dataset has only 90 clips. The result
replaces the drone, helicopter and background rows of configs/confusion/acoustic.json; the bird
and aircraft rows stay as documented assumptions (the dataset has no such audio).

False-alarm control: the raw classifier labels too many background windows as drone or
helicopter. On the out-of-fold predictions the script tries every combination of a confidence
threshold and a k-of-n vote over consecutive windows, and keeps the one with the best mean
drone/helicopter recall whose background false-alarm rate is at most --max-fa. The operating
point is chosen on the same out-of-fold predictions it is reported on, so the result is slightly
optimistic; the full table of operating points is saved for inspection.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing
import numpy as np

from nabhnetra.audio import make_model
from nabhnetra.common import ROOT
from nabhnetra.realdata import (AUDIO_LABELS, decide, kfold_by_group, load_wav, logmel,
                                parse_audio_name, windows)

BG = AUDIO_LABELS.index("background")
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
VOTES = [(1, 1), (2, 2), (2, 3), (3, 3), (3, 4)]


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


def train(X, y, epochs, seed, device):
    import torch

    torch.manual_seed(seed)
    model = make_model(len(AUDIO_LABELS)).to(device)
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


def predict_proba(model, X, device):
    import torch

    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X).to(device)), dim=1).cpu().numpy()


def confusion(y, pred):
    counts = np.zeros((len(AUDIO_LABELS), len(AUDIO_LABELS)), dtype=int)
    for t, p in zip(y, pred):
        counts[t, p] += 1
    return counts


def rates(counts):
    r = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return {"drone_recall": float(r[AUDIO_LABELS.index("drone"), AUDIO_LABELS.index("drone")]),
            "helicopter_recall": float(r[AUDIO_LABELS.index("helicopter"),
                                         AUDIO_LABELS.index("helicopter")]),
            "false_alarm": float(1.0 - r[BG, BG])}


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
    ap.add_argument("--max-fa", type=float, default=0.05,
                    help="highest acceptable share of background windows reported as a target")
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

    probs = np.zeros((len(y), len(AUDIO_LABELS)))
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        model = train(X[tr], y[tr], args.epochs, args.seed + k, device)
        probs[te] = predict_proba(model, X[te], device)
        print(f"  fold {k + 1}/{args.folds}: window accuracy "
              f"{(probs[te].argmax(1) == y[te]).mean():.3f}")

    raw_counts = confusion(y, probs.argmax(1))
    table = []
    for thr in THRESHOLDS:
        for k_, n_ in VOTES:
            c = confusion(y, decide(probs, groups, thr, k_, n_, BG))
            table.append({"threshold": thr, "k": k_, "n": n_, **rates(c), "counts": c.tolist()})
    ok = [r for r in table if r["false_alarm"] <= args.max_fa]
    if not ok:
        raise SystemExit(f"No operating point reaches false alarms <= {args.max_fa}")
    best = max(ok, key=lambda r: (r["drone_recall"] + r["helicopter_recall"]) / 2)
    counts = np.array(best["counts"])

    print("\nRaw classifier (argmax per window):", rates(raw_counts))
    print(f"Chosen rule: threshold {best['threshold']}, {best['k']} of last {best['n']} windows:",
          {k: round(v, 3) for k, v in rates(counts).items()})
    print("Rows true, columns reported:", AUDIO_LABELS)
    print(counts)
    measured = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)
    oof = ROOT / "runs" / "audio_oof.npz"
    oof.parent.mkdir(exist_ok=True)
    np.savez(oof, probs=probs, y=y, groups=groups)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    rows = dict(zip(data["true_classes"], data["matrices"]["all"]))
    rows["drone"] = measured[AUDIO_LABELS.index("drone")].round(4).tolist()
    rows["helicopter"] = measured[AUDIO_LABELS.index("helicopter")].round(4).tolist()
    data.update({
        "source": (f"measured {date.today().isoformat()}: {args.folds}-fold by file over "
                   f"{len(files)} clips ({len(y)} windows of {args.win} s), with the decision rule "
                   f"threshold {best['threshold']} and {best['k']}-of-{best['n']} vote chosen for "
                   f"false alarms <= {args.max_fa}. Drone, helicopter and background rows "
                   "measured; bird and aircraft rows assumed."),
        "labels": list(AUDIO_LABELS),
        "true_classes": list(rows),
        "matrices": {"all": list(rows.values())},
        "background_row": measured[AUDIO_LABELS.index("background")].round(4).tolist(),
        "assumed_rows": ["bird", "aircraft"],
        "counts_true_x_pred": counts.tolist(),
        "decision_rule": {"threshold": best["threshold"], "k": best["k"], "n": best["n"]},
        "raw_classifier": {**rates(raw_counts), "counts": raw_counts.tolist()},
        "operating_points": [{k: v for k, v in r.items() if k != "counts"} for r in table],
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
