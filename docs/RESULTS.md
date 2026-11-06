# Sensor ablation results (2026-11-06)

10 seeds per cell. Mean ± standard deviation. Metric definitions are in docs/DESIGN.md.

## mixed

_3 inbound drones, 12 birds (two flocks), 1 aircraft, 1 helicopter; clear day_

| Sensors | Drones identified % | Time to ID drone (s) | Track coverage % | Position RMSE (m) | False tracks / h | Class accuracy % | Bird as drone % | GOSPA (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radar | 37 ± 23 | 69 ± 74 | 68 ± 6 | 24.8 ± 2.8 | 2 ± 4 | 82 ± 4 | 2.4 ± 1.1 | 185 ± 13 |
| eo + ir | 73 ± 25 | 142 ± 36 | 4 ± 2 | 57.8 ± 3.3 | 1010 ± 423 | 96 ± 3 | 0.8 ± 2.1 | 338 ± 9 |
| acoustic | 0 ± 0 | n/a | 0 ± 0 | 82.4 ± 8.6 | 98 ± 31 | n/a | n/a | 286 ± 2 |
| radar + eo | 100 ± 0 | 35 ± 19 | 68 ± 6 | 22.2 ± 2.7 | 6 ± 10 | 91 ± 4 | 1.6 ± 1.1 | 180 ± 13 |
| radar + ir | 100 ± 0 | 18 ± 10 | 70 ± 6 | 19.4 ± 1.9 | 6 ± 10 | 96 ± 2 | 0.8 ± 0.8 | 175 ± 14 |
| radar + eo + ir | 100 ± 0 | 13 ± 8 | 70 ± 7 | 18.8 ± 1.9 | 4 ± 7 | 98 ± 1 | 0.5 ± 0.3 | 175 ± 15 |
| all | 100 ± 0 | 15 ± 9 | 70 ± 6 | 18.4 ± 2.0 | 0 ± 0 | 97 ± 1 | 0.9 ± 0.4 | 178 ± 14 |

![mixed ablation](media/ablation_mixed.png)

## night

_same traffic as mixed, at night (daylight camera nearly blind)_

| Sensors | Drones identified % | Time to ID drone (s) | Track coverage % | Position RMSE (m) | False tracks / h | Class accuracy % | Bird as drone % | GOSPA (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radar | 37 ± 23 | 69 ± 74 | 68 ± 6 | 24.8 ± 2.8 | 2 ± 4 | 82 ± 4 | 2.4 ± 1.1 | 185 ± 13 |
| eo + ir | 73 ± 25 | 155 ± 22 | 4 ± 2 | 57.1 ± 2.9 | 831 ± 317 | 95 ± 4 | 1.3 ± 2.7 | 326 ± 7 |
| acoustic | 0 ± 0 | n/a | 0 ± 0 | 82.4 ± 8.6 | 98 ± 31 | n/a | n/a | 286 ± 2 |
| radar + eo | 47 ± 27 | 71 ± 48 | 67 ± 7 | 24.6 ± 2.9 | 2 ± 4 | 82 ± 4 | 2.2 ± 1.4 | 185 ± 14 |
| radar + ir | 100 ± 0 | 18 ± 10 | 70 ± 6 | 19.4 ± 1.9 | 6 ± 10 | 96 ± 2 | 0.8 ± 0.8 | 175 ± 14 |
| radar + eo + ir | 100 ± 0 | 15 ± 9 | 69 ± 7 | 19.0 ± 2.1 | 2 ± 4 | 96 ± 2 | 0.7 ± 0.7 | 175 ± 15 |
| all | 100 ± 0 | 19 ± 10 | 69 ± 6 | 19.5 ± 1.6 | 2 ± 4 | 97 ± 2 | 1.1 ± 0.9 | 179 ± 14 |

![night ablation](media/ablation_night.png)

## fog

_same traffic as mixed, 800 m visibility_

| Sensors | Drones identified % | Time to ID drone (s) | Track coverage % | Position RMSE (m) | False tracks / h | Class accuracy % | Bird as drone % | GOSPA (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radar | 37 ± 23 | 69 ± 74 | 68 ± 6 | 24.8 ± 2.8 | 2 ± 4 | 82 ± 4 | 2.4 ± 1.1 | 185 ± 13 |
| eo + ir | 17 ± 22 | 169 ± 20 | 1 ± 1 | 60.7 ± 7.0 | 298 ± 377 | 97 ± 3 | 0.0 ± 0.0 | 291 ± 10 |
| acoustic | 0 ± 0 | n/a | 0 ± 0 | 82.4 ± 8.6 | 98 ± 31 | n/a | n/a | 286 ± 2 |
| radar + eo | 60 ± 29 | 123 ± 62 | 68 ± 6 | 25.0 ± 2.8 | 2 ± 4 | 83 ± 4 | 2.3 ± 1.3 | 185 ± 13 |
| radar + ir | 90 ± 15 | 109 ± 36 | 67 ± 6 | 24.2 ± 2.7 | 2 ± 4 | 85 ± 4 | 2.1 ± 1.4 | 184 ± 14 |
| radar + eo + ir | 93 ± 13 | 93 ± 51 | 69 ± 6 | 24.8 ± 3.1 | 3 ± 6 | 85 ± 4 | 1.8 ± 0.9 | 183 ± 14 |
| all | 97 ± 10 | 108 ± 54 | 68 ± 6 | 24.6 ± 2.9 | 2 ± 4 | 85 ± 4 | 2.9 ± 1.2 | 188 ± 14 |

![fog ablation](media/ablation_fog.png)

## bird_flock

_2 drones approaching through 4 bird flocks (22 birds) on their paths_

| Sensors | Drones identified % | Time to ID drone (s) | Track coverage % | Position RMSE (m) | False tracks / h | Class accuracy % | Bird as drone % | GOSPA (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radar | 40 ± 30 | 84 ± 56 | 45 ± 4 | 23.9 ± 2.4 | 3 ± 6 | 86 ± 5 | 3.0 ± 2.0 | 269 ± 7 |
| eo + ir | 85 ± 23 | 149 ± 29 | 4 ± 1 | 59.1 ± 2.0 | 1077 ± 259 | 93 ± 3 | 1.0 ± 0.6 | 410 ± 5 |
| acoustic | 0 ± 0 | n/a | 0 ± 0 | n/a | 32 ± 23 | n/a | n/a | 346 ± 1 |
| radar + eo | 100 ± 0 | 48 ± 32 | 46 ± 3 | 23.3 ± 2.7 | 2 ± 4 | 93 ± 3 | 1.0 ± 0.7 | 267 ± 7 |
| radar + ir | 100 ± 0 | 26 ± 21 | 46 ± 3 | 22.8 ± 2.4 | 2 ± 4 | 96 ± 2 | 0.7 ± 0.6 | 269 ± 7 |
| radar + eo + ir | 100 ± 0 | 19 ± 10 | 46 ± 4 | 22.3 ± 2.7 | 6 ± 10 | 97 ± 1 | 0.6 ± 0.4 | 271 ± 7 |
| all | 100 ± 0 | 21 ± 16 | 46 ± 4 | 22.4 ± 2.9 | 3 ± 6 | 96 ± 2 | 0.8 ± 0.4 | 270 ± 8 |

![bird_flock ablation](media/ablation_bird_flock.png)

## swarm

_20 drones from one 30-degree sector plus 4 birds_

| Sensors | Drones identified % | Time to ID drone (s) | Track coverage % | Position RMSE (m) | False tracks / h | Class accuracy % | Bird as drone % | GOSPA (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radar | 38 ± 11 | 86 ± 16 | 90 ± 2 | 22.4 ± 0.9 | 3 ± 6 | 42 ± 8 | 1.3 ± 2.6 | 139 ± 6 |
| eo + ir | 99 ± 2 | 53 ± 9 | 7 ± 2 | 67.1 ± 1.1 | 1820 ± 226 | 94 ± 3 | 10.2 ± 5.7 | 418 ± 3 |
| acoustic | 12 ± 6 | 190 ± 16 | 0 ± 0 | 76.1 ± 7.3 | 348 ± 76 | n/a | n/a | 350 ± 4 |
| radar + eo | 100 ± 0 | 28 ± 4 | 90 ± 2 | 21.0 ± 0.9 | 8 ± 8 | 78 ± 4 | 1.8 ± 2.9 | 136 ± 6 |
| radar + ir | 100 ± 0 | 17 ± 4 | 91 ± 2 | 19.6 ± 1.4 | 4 ± 7 | 91 ± 1 | 0.9 ± 1.9 | 131 ± 13 |
| radar + eo + ir | 100 ± 0 | 17 ± 5 | 91 ± 2 | 18.9 ± 0.8 | 6 ± 12 | 94 ± 1 | 0.5 ± 1.1 | 136 ± 10 |
| all | 100 ± 0 | 13 ± 3 | 91 ± 2 | 18.4 ± 1.3 | 8 ± 10 | 94 ± 1 | 2.0 ± 1.3 | 135 ± 10 |

![swarm ablation](media/ablation_swarm.png)

## radar_failure

_same traffic as mixed; radar is down from 80 s to 160 s_

| Sensors | Drones identified % | Time to ID drone (s) | Track coverage % | Position RMSE (m) | False tracks / h | Class accuracy % | Bird as drone % | GOSPA (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radar | 33 ± 26 | 54 ± 64 | 43 ± 4 | 25.5 ± 2.8 | 20 ± 12 | 82 ± 4 | 1.9 ± 1.1 | 221 ± 10 |
| eo + ir | 73 ± 25 | 142 ± 36 | 4 ± 2 | 57.8 ± 3.3 | 1010 ± 423 | 96 ± 3 | 0.8 ± 2.1 | 338 ± 9 |
| acoustic | 0 ± 0 | n/a | 0 ± 0 | 82.4 ± 8.6 | 98 ± 31 | n/a | n/a | 286 ± 2 |
| radar + eo | 100 ± 0 | 53 ± 52 | 45 ± 4 | 25.1 ± 2.3 | 6 ± 10 | 91 ± 3 | 1.3 ± 0.8 | 221 ± 9 |
| radar + ir | 100 ± 0 | 18 ± 10 | 46 ± 5 | 22.8 ± 2.0 | 21 ± 10 | 96 ± 1 | 0.7 ± 0.7 | 220 ± 11 |
| radar + eo + ir | 100 ± 0 | 13 ± 8 | 47 ± 4 | 23.0 ± 2.3 | 4 ± 10 | 97 ± 1 | 0.6 ± 0.4 | 222 ± 10 |
| all | 100 ± 0 | 15 ± 9 | 49 ± 4 | 22.2 ± 2.0 | 0 ± 0 | 97 ± 1 | 0.8 ± 1.0 | 221 ± 8 |

![radar_failure ablation](media/ablation_radar_failure.png)

## Detectors by the dataset's own distance tags (real held-out footage)

### Thermal (IR)

| Distance bin | Videos | Frames | Target found | Right class (of found) | Found, by class |
|---|---:|---:|---:|---:|---|
| close | 11 | 693 | 98% | 100% | drone 99%, aircraft 100%, helicopter 98% |
| medium | 29 | 1759 | 97% | 93% | drone 100%, bird 85%, aircraft 100%, helicopter 97% |
| distant | 29 | 1791 | 93% | 92% | drone 95%, bird 98%, aircraft 93%, helicopter 27% |

### Daylight (EO)

| Distance bin | Videos | Frames | Target found | Right class (of found) | Found, by class |
|---|---:|---:|---:|---:|---|
| close | 12 | 705 | 96% | 99% | drone 100%, bird 26%, aircraft 94%, helicopter 97% |
| medium | 29 | 1805 | 97% | 97% | drone 99%, bird 89%, aircraft 97%, helicopter 99% |
| distant | 14 | 881 | 97% | 94% | drone 99%, bird 97%, aircraft 93%, helicopter 98% |
