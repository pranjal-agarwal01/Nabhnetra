# Released model weights

These are the trained models behind every result in the repository, so the demos run straight
after cloning. Scripts use a freshly trained model in `runs/` or `models/` if one exists, and
these files otherwise. Details of each model are in [docs/MODEL_CARD.md](../docs/MODEL_CARD.md).

| File | Model | Trained on | Held-out result | Size | Licence |
|---|---|---|---|---|---|
| `thermal_yolo11n.pt` | YOLO11n detector, 4 classes | 287 thermal videos | mAP50 0.825 on 72 unseen videos | 5.5 MB | AGPL-3.0 |
| `daylight_yolo11n.pt` | YOLO11n detector, 4 classes | 218 daylight videos | mAP50 0.850 on 55 unseen videos | 5.5 MB | AGPL-3.0 |
| `audio_cnn.pt` | 3-block CNN on log-mel spectrograms | 90 audio clips | 5-fold by clip: drone recall 80% at 4% false alarms (with the decision rule) | 0.1 MB | MIT |

The two detectors were fine-tuned from Ultralytics YOLO11n, so they are distributed under the
AGPL-3.0 licence of Ultralytics, not the MIT licence of the rest of the repository. The audio
model is this project's own architecture, trained with plain PyTorch, and is MIT like the code.

All three were trained on the multi-sensor drone detection dataset of F. Svanström,
F. Alonso-Fernandez and C. Englund (*Data in Brief* 39, 2021; CC BY 4.0).

## Checksums (SHA-256)

```
548b2952a23e27dced487b2bef9dc1342b6c2da4324ceaf046903f8d09e102d8  thermal_yolo11n.pt
e32041d2244413962be124466ec3e2e26cbcae8dc3595d9df77a456b0bdd2f8b  daylight_yolo11n.pt
9fc2096eff6805a33da539b6a7a8c4b7dcb49bd0096ce2800b34539a8c2b5083  audio_cnn.pt
```

Loading a model file runs Python's unpickler, so only load weights from a source you trust and
check the checksum first.
