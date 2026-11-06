# Speed (6 Nov 2026)

Measured with `scripts/benchmark.py` on a laptop with an NVIDIA RTX 4060 Laptop GPU and an
Intel Core (family 6, model 186) CPU, in three separate runs. Laptop timings varied by up to 4x
between runs (power and thermal state), so ranges are given rather than a single best case.

| Part | Hardware | Time per frame or step | Rate |
|---|---|---:|---:|
| Thermal detector (YOLO11n, 640 px) | RTX 4060 Laptop GPU | 8–22 ms | 45–120 frames/s |
| Daylight detector (YOLO11n, 640 px) | RTX 4060 Laptop GPU | 8–30 ms | 33–125 frames/s |
| Either detector | Laptop CPU only | 220–280 ms | 3.6–4.6 frames/s |
| Fusion step, mixed traffic (17 targets, all sensors, IMM) | One CPU core | 16–72 ms mean | 7–31x faster than real time |
| Fusion step, 20-drone swarm (24 targets) | One CPU core | 43–88 ms mean | 6–12x faster than real time |

Each fusion step covers 0.5 s of sensor data, so any step under 500 ms keeps up with real time.
Half precision (FP16) made no difference on the GPU: for a model this small, pre-processing and
data transfer dominate.

What this means:
- With a GPU, both detectors and the fusion run comfortably in real time on one laptop.
- Without a GPU, detection runs at about 4 frames per second. A CPU-only deployment would need
  a lower input resolution, frame skipping, or an optimised runtime (ONNX Runtime, OpenVINO).
- The fusion itself is cheap, even with the IMM tracker and a 20-drone swarm.
