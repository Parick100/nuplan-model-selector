# Second Project Report

**Which Model When: A Controlled Comparison of Regression, Diffusion and Flow Matching for Vehicle Trajectory Prediction**

Members: Member A, Member B, Member C, Member D · Code: https://github.com/`<username>`/`<repo>`

## 1. Data

nuPlan `mini` v1.1 — 64 driving logs from four cities (Las Vegas, Singapore, Boston,
Pittsburgh). Task: agent-centric prediction of the next 8 s (80 points at 10 Hz) from
2 s of target history, expressed in the target's frame at t₀. Each sample stores
`hist` (20×4: x, y, vx, vy), `neigh` (10×20×6 for up to ten nearby agents),
`neigh_mask`, `future` (80×2) and 67 multi-hot `scenario_tag`s. After filtering
near-stationary windows: **297,875 samples**, split **by log and stratified by city**
into 175,522 train / 61,327 selector-labelling / 61,026 held-out test. A four-stage
pipeline builds memory-mapped caches and per-element normalisation, and an automated
checker verifies shapes, split disjointness and city coverage.

## 2. Models

Three paradigms share one encoder (1.23 M) and one DiT decoder (2.47 M) — **identical
3,694,754 parameters per arm**, so the comparison isolates the generative mechanism:
(A) deterministic regression, 1 forward; (B) diffusion, VP-SDE, 15-step sampler,
90 NFE; (C) flow matching on a linear path, 4-step sampler, 24 NFE (K = 6). All three
predict x₀ rather than ε/v, are constrained to a 48-dimensional **velocity-space DCT
subspace**, and use a metric-weighted loss so every timestep contributes equally in
metres. Regression is exactly the t = 0, x = 0 corner of both generative models.

## 3. Current progress

Held-out split, 4,096 scenes, K = 6, 3 seeds:

| model | NFE | minADE | minFDE | miss@2m | latency |
|---|---|---|---|---|---|
| constant velocity | 1 | 5.937 | 16.882 | 0.941 | — |
| A regression | 1 | **2.823** | 8.252 | 0.837 | **6.2 ms** |
| C flow matching | 24 | **1.826** | 4.825 | 0.574 | 18.9 ms |
| B diffusion | 90 | **1.736** | 4.525 | 0.580 | 59.5 ms |

Generative models beat regression by 1.5–1.6×, and all beat constant velocity by
3–3.4×. **RQ2 answer:** diffusion wins by only **4.9 %** at **3.75×** the compute;
flow matching saturates at 4 steps, and its 1-step form coincides with regression
(2.943 vs 2.944).

For the selector we implemented oracle labelling, cascaded gates and a cost-sensitive
regressor. Key methodological finding: the *naive* oracle gap is **22.4 %**, but a
**split-half honest oracle** — select the model on half of each model's candidates and
evaluate on the other half — retains only **4.7 %**, so ≈80 % of the naive gap is
selection noise rather than attainable gain. Our learned selector recovers
**+1.1 % to +1.7 %** over the linear interpolation of the fixed arms at matched
latency, i.e. about a third of the honest ceiling.

Correctness is verified rather than assumed: a floor-relative single-scene overfit test
(with a control run that correctly fails) and an *oracle-sampler* test whose error is
**0.000e+00** for 1…256 sampling steps in both paradigms.

## 4. Problems encountered

**Data.** Absolute UTM coordinates caused a 0.25 m float32 quantisation (now
3.8·10⁻⁶ m). Heading had to come from `lidar_box.yaw` rather than a position
difference. Track positions carry frame-level jitter, so ground truth violates *any*
absolute acceleration threshold 100 % of the time; the kinematic metric was therefore
made relative to the data's own distribution.

**Modelling.** ε-parameterisation amplified field error up to 152×. A *position*-space
DCT constraint added a systematic 0.2744·(v·dt) endpoint bias that its "99.9996 %
energy retained" statistic completely hid (0.41 m at 15 m/s); it was replaced by the
velocity-space form. Our first comparison was also unfair (a 284 K MLP head against a
2.47 M DiT) and was corrected. Finally, best-of-K (WTA), the standard remedy for
multimodal collapse, **fails** for sampler-integrated fields (2.539 vs 1.757 minADE).

**Remaining limitations (measured).** Output is over-smoothed (roughness 0.07× ground
truth); K = 6 candidates span only 2.2–2.65 m laterally against a ground-truth p95 of
19.7 m; sharp turns above 10 m/s are essentially missed; there is no map input yet.

## 5. Plan for the next steps

1. Close the selector gap with map/lane features and direct optimisation of the
   accuracy–latency Pareto front instead of threshold labels.
2. Add lane geometry to the encoder — the root cause of the turn failures.
3. A sampling-compatible mode-seeking objective (CFG-style guidance or a repulsion
   term across candidates), since WTA is ruled out.
4. Multi-seed repeats of the full comparison and per-scenario stratification.
