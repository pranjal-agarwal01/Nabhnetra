# Sample clips

Eight 10-second clips from the Svanström multi-sensor drone dataset, one per class for each
camera. Every clip is from the **validation split**, so the detectors never trained on any frame
of it. Each comes with its `_LABELS.mat` ground-truth file, so the detection script can score it.

| Thermal (IR) | Daylight (V) |
|---|---|
| IR_DRONE_001 | V_DRONE_026 |
| IR_BIRD_020 | V_BIRD_028 |
| IR_AIRPLANE_006 | V_AIRPLANE_009 |
| IR_HELICOPTER_026 | V_HELICOPTER_007 |

Run the detectors on all of them:

```bash
python -I scripts/detect_video.py samples
```

The dataset's own demo clips (`Demo_IR_DRONE_146`, `Demo_V_DRONE_017`) are **training** videos,
so they are not used here.

Source: F. Svanström, F. Alonso-Fernandez, C. Englund, "A dataset for multi-sensor drone
detection", *Data in Brief* 39 (2021). Licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Files copied unchanged.
