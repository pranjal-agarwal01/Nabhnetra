"""Fine-tune a YOLO detector on the prepared dataset (one model per camera).

    python scripts/train_detector.py --data data/yolo_ir/dataset.yaml --name ir
    python scripts/train_detector.py --data data/yolo_v/dataset.yaml --name eo

GPU: training needs a CUDA build of torch (check https://pytorch.org for the current command), e.g.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
The published models were trained on an RTX 4060 laptop GPU, about an hour per camera.

Licence note: Ultralytics is AGPL-3.0. The rest of Nabhnetra only consumes a confusion matrix,
so the detector can be swapped for an Apache-2.0 model (YOLOX, RT-DETR) without other changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset.yaml written by prepare_svanstrom.py")
    ap.add_argument("--name", required=True, help="run name, e.g. ir or eo")
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (default yolo11n.pt)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10, help="stop early after this many flat epochs")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu to force CPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    from nabhnetra.realdata import disable_ultralytics_analytics

    disable_ultralytics_analytics()
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Training on CPU. This will be slow; see the GPU note at the top of this script.")
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                batch=args.batch, workers=args.workers, device=device, project="runs/detect",
                name=args.name, exist_ok=True, seed=args.seed, deterministic=True, plots=True)
    print(f"\nBest weights: runs/detect/{args.name}/weights/best.pt")
    print("Next: python scripts/eval_confusion.py --weights "
          f"runs/detect/{args.name}/weights/best.pt --data <dataset folder> "
          f"--out configs/confusion/{args.name}.json")


if __name__ == "__main__":
    main()
