# Design notes

The models, equations and assumptions behind each module.

## Frames and conventions

- Local east-north-up (ENU) frame in metres, origin at the defended asset.
- Azimuth is clockwise from north; elevation is above the horizon.
- Time step `dt = 0.5 s`. Sensors report at their own period (radar 1 s, cameras 0.5 s,
  acoustic 1 s, ADS-B 1 s).

## Sensor models (`nabhnetra/sensors.py`)

### Radar
- Measures `z = [range, azimuth, elevation] + noise`. Defaults: σ = 5 m, 0.5°, 0.8°.
- Detection follows the radar range equation with a Swerling-I target:
  `SNR = SNR50 · (σ_rcs / σ_ref) · (R50 / r)^4` and `Pd = Pfa^(1 / (1 + SNR))`.
  `SNR50` is chosen so a 0.01 m² target is detected half the time at `R50` = 3 km.
- Small drones (0.005–0.05 m²) and birds (0.001–0.02 m²) have similar radar cross-sections.
  That overlap is why radar alone cannot tell them apart.
- Targets in the same resolution cell (15 m in range, 1.5° in azimuth) merge into one echo at
  the RCS-weighted centre, which is how a bird flock looks to a real radar.
- Clutter: Poisson false alarms per scan, uniform over area, at low elevation.

### EO and IR cameras
- Measure `z = [azimuth, elevation] + noise` with no range. Defaults: σ = 0.3 mrad (EO) and
  0.5 mrad (IR), about 30 times finer than the radar's angle.
- Pixels on target: `px = size / (range · IFOV)`.
- Detection: `Pd = logistic((px − px50) / scale) · night_factor · exp(−3.912 · r / (V · gain))`.
  The last term is Koschmieder's contrast law for visibility `V`; thermal IR gets `gain = 3`.
- Classification: the label is sampled from the confusion matrix for the current size bin
  (distant < 8 px ≤ medium < 20 px ≤ close). These are the same bins `eval_confusion.py`
  measures on real data.
- Label persistence: with probability 0.7 the previous label repeats. Long-run rates still
  match the matrix, but errors come in runs, as real detector errors do.

### Acoustic array
- Measures direction with σ = 4° azimuth and 8° elevation.
- Detection: `SNR = SL − 20·log10(r) − α·r − noise + array_gain`, passed through a logistic.
  Birds are effectively silent, so hearing something is itself evidence against "bird".
- The classifier outputs drone, helicopter or background. A "background" output means no report.

### ADS-B
- Cooperative aircraft report their own position (σ = 10 m horizontal, 15 m vertical) and an
  identity. Spoofing is not modelled, so cooperative status is treated as a hint, not proof.

## Tracker (`nabhnetra/tracker.py`)

- State `x = [e, n, u, ve, vn, vu]`, nearly-constant-velocity model:
  `F = [[I, dt·I], [0, I]]`, `Q = q · [[dt³/3·I, dt²/2·I], [dt²/2·I, dt·I]]`, with q = 6 m²/s³.
- Measurement models: linear for ADS-B; extended Kalman filter for radar and cameras, with the
  Jacobian of `[r, az, el]` with respect to position in `geometry.rae_jacobian` (unit-tested
  against finite differences). Azimuth residuals are wrapped to [−π, π).
- Gating: Mahalanobis distance `d² = yᵀ S⁻¹ y` under the χ² 99.7% quantile for the measurement
  dimension.
- Association: global nearest neighbour. The Hungarian algorithm minimises
  `d² + ln|S|` (the negative log-likelihood up to a constant) per sensor scan.
- Update: Joseph form `P = (I−KH) P (I−KH)ᵀ + K R Kᵀ`, which keeps P symmetric positive definite.
- Initiation: a radar plot or ADS-B report starts a tentative track. Position covariance is the
  measurement noise mapped through the polar-to-Cartesian Jacobian; velocity σ is 50 m/s.
- Confirmation: 3 hits. **Direction-only reports may not update tentative tracks while radar is
  active.** That rule stops clutter-initiated tracks from being confirmed by a camera that sees
  a real target on a similar bearing (ghosting).
- Deletion: tentative after 2.5 s without an update; confirmed after 5 s, or when the position
  σ exceeds 1.5 km.
- Duplicate merge: tracks within 25 m and 5 m/s of each other are merged, keeping the one with
  more hits.
- Camera-only configurations start tracks at an assumed 1 km range with σ = 600 m. From a single
  site the range never becomes observable, and the ablation shows that honestly.

## Identity fusion (`nabhnetra/identity.py`)

For each track and class c:

    E_c(t) = exp(−Δt/τ) · E_c(t_prev) + w_sensor · log L_c        (L = confusion-matrix column)
    P(c)   = softmax(log prior_c + E_c + K_c)

- τ = 10 s and w_sensor < 1 temper correlated evidence (naive Bayes over many frames of the
  same blurry target is overconfident).
- `K_c` is a kinematic term: the track's speed under each class's speed model, with the track's
  own velocity uncertainty added. It is recomputed, never accumulated.
- Each sensor's share of the evidence is kept for the operator's "why" panel.

## Threat ranking (`nabhnetra/threat.py`)

    danger = 1 − (1 − proximity) · (1 − urgency · heading)
    score  = class_weight · danger · cooperative_factor

proximity = exp(−r / 1.5 km); urgency = 1 / (1 + time_to_go / 60 s) when closing;
heading = 1.0 if the closest approach is under 300 m, 0.6 under 1 km, else 0.3.
Levels: HIGH ≥ 0.6, MEDIUM ≥ 0.3, LOW ≥ 0.1. The score is advisory and nothing acts on it.

## Metrics (`nabhnetra/metrics.py`)

Truth inside a 4 km radius is evaluated at every step.

| Metric | Definition |
|---|---|
| GOSPA (m) | α = 2, p = 2, c = 100 m. Combines localisation error, missed targets and false tracks |
| Position RMSE | Over truth–track pairs matched by GOSPA (closer than c) |
| Track coverage | Matched truth-steps divided by truth-steps |
| False tracks per hour | Confirmed tracks never matched to any truth |
| Class accuracy | Fused label equals the true class, over matched pairs |
| Bird as drone | Share of matched bird pairs whose fused label is drone |
| Drones identified | Drones whose track reached P(drone) ≥ 0.6 at least once |
| Time to identify | From the drone entering the volume to that moment (median) |

## Known limitations

- Radar, ADS-B and the scenarios are simulated. Real radar data with matching video would
  replace the radar model.
- The dataset's drones were filmed within about 200 m: target boxes have a median size of 17 px
  (IR) and 35 px (visible), so almost nothing falls in the "distant" (< 8 px) bin.
  `eval_confusion.py` fills the small-size bins by shrinking validation frames to 1/2 and 1/4
  (range extension). That ignores atmosphere and optics, so long-range figures are approximate.
- Label quality (checked 6 Nov 2026): all 650 label files decode, but 18 videos (mostly birds)
  have label counts that do not match their frame counts, because frames with two boxes are
  skipped by the decoder. `prepare_svanstrom.py` excludes them. Visible-camera boxes are loose
  (about 1.5–3× the target), so pixel sizes measured from them overstate the target.
- There is one sensor site, so cameras alone cannot measure range.
- With a single constant-velocity model, sharp manoeuvres are followed through process noise
  rather than a manoeuvre model (IMM is on the roadmap).
- The label decoder keeps one box per frame, so frames with several birds are under-labelled.
