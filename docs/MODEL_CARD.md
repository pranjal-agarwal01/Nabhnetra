# Model cards

Three trained models feed Nabhnetra. Each card says what the model is for, what it was trained
and tested on, how well it does, and where it fails. Weights are in [weights/](../weights/).

## Thermal and daylight detectors

| | Thermal | Daylight |
|---|---|---|
| File | `weights/thermal_yolo11n.pt` | `weights/daylight_yolo11n.pt` |
| Architecture | YOLO11n (Ultralytics), 2.6 M parameters, fine-tuned from COCO weights | Same |
| Input | One thermal frame (dataset frames are 320×256), resized to 640 px | One daylight frame (640×512) |
| Output | Boxes with class (drone, bird, aircraft, helicopter) and confidence | Same |
| Training | 287 videos, 18,058 frames (every 5th), 23 epochs, early-stopped (best at 13) | 218 videos, 13,735 frames, 25 epochs (best at 15) |
| Validation | 72 unseen videos, 4,543 frames | 55 unseen videos, 3,466 frames |
| Precision / recall | 0.833 / 0.819 | 0.856 / 0.825 |
| mAP50 / mAP50-95 | 0.825 / 0.435 | 0.850 / 0.406 |
| Best / weakest class (mAP50) | drone 0.909 / bird 0.755 | helicopter 0.894 / bird 0.822 |
| Found at full / half / quarter frame size | 95.5% / 84.0% / 9.5% | 96.8% / 56.1% / 13.1% |
| 50% detection point | about 7.5 px on target | about 18.5 px (loose boxes; true target smaller) |
| Licence | AGPL-3.0 (Ultralytics) | AGPL-3.0 (Ultralytics) |

**Intended use.** Research and education in multi-sensor fusion: producing detections whose
measured behaviour drives the simulation, and demonstrating detection on recorded footage.

**Out of scope.** Operational air defence or any decision about people or property. The models
have not been tested outside the training dataset's conditions.

**Data.** Svanström multi-sensor drone detection dataset (Halmstad University, Sweden; CC BY
4.0): daylight recordings at Swedish airports and fields, drones within about 200 m (DJI Phantom 4
Pro, DJI F450, Hubsan H107D+), plus birds, airplanes and helicopters; 49 aircraft videos come
from one YouTube channel. Splits are by whole video and stratified by class. 18 videos whose
label counts do not match their frame counts were excluded.

**Known failure modes.**
- Distant aircraft in thermal look like birds (a 4–5 pixel warm streak); in one sample clip the
  class flips between aircraft and bird.
- Distant helicopters in thermal are often missed (27% found in the "distant" bin).
- Close-up birds in daylight are often missed (26% found in the "close" bin; few training examples).
- Small targets are boxed loosely, so mAP50-95 is much lower than mAP50.
- Many unlabelled frames in bird videos contain real birds, so the false-alarm rate on
  "empty" frames (0.6–0.8 per frame) is an upper bound, not a measurement.

**Not yet known.** Performance on Indian sites, haze, dust, heat shimmer, local bird species
(black kites, vultures, egrets) and other drone types; behaviour beyond the filmed range except
as estimated by shrinking frames.

## Audio classifier

| | |
|---|---|
| File | `weights/audio_cnn.pt` |
| Architecture | 3 convolution blocks (16, 32, 64 channels) on 64-band log-mel spectrograms, global average pooling, linear layer; this project's own model |
| Input | 1 s of mono audio at 22,050 Hz; windows every 0.5 s |
| Output | Probabilities for drone, helicopter, background |
| Training | 90 ten-second clips (30 per class), SpecAugment masking, 25 epochs |
| Evaluation | 5-fold cross-validation by whole clip (1,710 windows) |
| Raw result | Drone recall 96%, helicopter 93%, background false alarms 31% |
| With decision rule | Confidence ≥ 0.9 in each of the last 3 windows: drone 80%, helicopter 70%, false alarms 4% |
| Licence | MIT |

**Caveats.** The decision rule was chosen on the same out-of-fold predictions it is reported on,
so its figures are slightly optimistic. The dataset has no bird or airplane audio; those rows of
the simulator's audio matrix are assumptions. The saved weights were retrained on all 90 clips,
so testing them on the dataset's own clips does not measure accuracy.

## Responsible use

All three models feed a decision-support system for human operators. Nabhnetra contains no
engagement or weapon-control logic, and its threat levels are advisory.
