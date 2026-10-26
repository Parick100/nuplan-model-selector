# Which Model When — Generative Vehicle Trajectory Prediction

A **controlled comparison** of three trajectory-generation paradigms that share one
encoder *and one decoder* — deterministic regression, diffusion, and flow matching —
plus (planned) a cheap runtime **model selector** that picks the cheapest model that
still meets a quality budget.

Course project, *Introduction of Machine Learning* (Term 5). Dataset: **nuPlan mini**.

---

## 1. The two research questions

1. **QR1** — How much accuracy / diversity does generative modelling buy over a
   deterministic regression baseline in multimodal driving scenes?
2. **QR2** — Can flow matching reach diffusion-level quality with substantially
   fewer sampling steps (i.e. less compute)?

The project's deliverable is *not* a single model but a **decision rule**: given a
compute budget, which of the three models should be run? That requires the three arms
to be strictly comparable, which is what most of the engineering below is about.

---

## 2. Headline results

Held-out split (13 logs never seen by any model), 4,096 scenes, 3 sampling seeds,
K = 6 candidates, all metrics in **metres**.

| Model | NFE | minADE@6 | minFDE@6 | miss@2m | diversity | latency (batch 1) | params |
|---|---|---|---|---|---|---|---|
| Constant velocity (baseline) | 1 | 5.937 | 16.882 | 0.941 | — | — | — |
| **A** regression | 1 | **2.823 ± 0.000** | 8.252 | 0.837 | — | **6.2 ms** | 3.69 M |
| **C** flow matching (4 steps) | 24 | **1.826 ± 0.011** | 4.825 | 0.574 | 7.5 | 18.9 ms | 3.69 M |
| **B** diffusion (15 steps) | 90 | **1.736 ± 0.017** | 4.525 | 0.580 | 9.3 | 59.5 ms | 3.69 M |

**Answer to QR2 (so far): yes.** Diffusion beats flow matching by only **4.9 %**
(1.736 vs 1.826) while costing **3.75×** the network evaluations (90 vs 24) and
**3.1×** the wall-clock time. A step sweep shows flow matching saturates at **4 steps**:

| steps | 1 | 2 | **4** | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| flow-matching minADE | 2.943 | 2.237 | **2.098** | 2.089 | 2.089 | 2.095 |

A useful consistency check: **1-step flow matching = 2.943 vs regression = 2.944**.
At one step `x_{t'=0} = x̂₀`, so flow matching degenerates algebraically into
regression — and it does, to three digits.

---

## 3. Data

**Source.** nuPlan `mini` split, v1.1 (64 `.db` SQLite logs, 4 cities:
las_vegas / singapore-one-north / boston / pittsburgh-hazelwood).

**Task.** Given 2 s of history at 10 Hz, predict 8 s of future at 10 Hz (80 points),
expressed in the target agent's frame at t0 (origin, heading = +x).

**Tensors per sample** (frozen contract, verified by `code/data/verify_cache.py`):

| key | shape | meaning |
|---|---|---|
| `hist` | (20, 4) | target: x, y, vx, vy |
| `neigh` | (10, 20, 6) | up to 10 neighbours: x, y, vx, vy, w, l |
| `neigh_mask` | (10,) | slot validity |
| `future` | (80, 2) | ground-truth future |
| `tags` | (67,) | multi-hot `scenario_tag` |
| `t0`, `track_id` | scalar | provenance |

**Scale.** 297,875 windows after filtering; split **by log and stratified by city**
into model-train / selector-labelling / selector-test:

| split | samples | role |
|---|---|---|
| 0 | 175,522 | model training |
| 1 | 61,327 | oracle labelling + selector training |
| 2 | 61,026 | selector test (never used for training or tuning) |

**Pipeline.** `code/data/extract.py` (single log) → `build_mini.py` (multi-log,
memory-safe) → `convert_cache.py` (npz → mmap `.npy`) → `make_norm.py` (per-element
`mu`/`sd`) → `dataset.py` (mmap dataset).

**Data-quality findings** (each one caused a real bug before being fixed):

* nuPlan stores **absolute UTM** coordinates (`|y| ≈ 4·10⁶`); differencing them in
  float32 introduces a 0.25 m quantisation. Fixed by forcing float64 in
  `to_target_frame`; independently re-verified, residual now **3.8·10⁻⁶ m**
  (= float32 storage precision) vs 0.25 m before.
* Heading must come from `lidar_box.yaw`, not from a 0.1 s position difference
  (the latter had a 66° median error on 15.7 % of samples).
* Track positions carry **frame-level jitter**: measured `corr(Δp/dt, db velocity)
  = 0.993`, so the jitter is in the tracker output itself. Consequence: ground truth
  violates any *absolute* acceleration threshold 100 % of the time, so the kinematic
  metric is reported **relative to the ground truth's own pointwise p95**
  (ground-truth violation rate is 0.05 by construction, not 1.000).

---

## 4. Models — identical capacity on purpose

All three arms share the encoder **and the decoder**. The only differences are the
training objective and the sampler.

| | A regression | C flow matching | B diffusion |
|---|---|---|---|
| encoder | 1,228,864 | 1,228,864 | 1,228,864 |
| decoder (DiT) | 2,465,890 | 2,465,890 | 2,465,890 |
| **total** | **3,694,754** | **3,694,754** | **3,694,754** |
| output DOF | 48 | 48 | 48 |
| training target | x₀ | x₀ | x₀ |
| sampler | 1 forward | x₀-DDIM, 4 steps | x₀-DDIM, 15 steps |
| NFE (K=6) | 1 | 24 | 90 |

**Regression is the t=0, x=0 corner of both generative models.** There `x_t` carries
no information about `x₀`, so the model must rely purely on the context — which is the
definition of deterministic regression. Hence one `predict_field` serves all three arms.

Three design decisions worth explaining:

1. **All objectives predict x₀, not ε / v.** A λ-mixing experiment showed that even
   with field error 1e-4, ε-parameterisation is 7× worse than flow matching, because
   recovering `x̂₀ = (x_t − √(1−ᾱ)ε̂)/√ᾱ` amplifies field error by
   `√((1−ᾱ)/ᾱ)` — 152× at t=1. With x₀ everywhere, the only remaining difference
   between the paradigms is the probability path itself.
2. **Fixed DCT low-pass constraint on the *velocity* profile.** The output head emits
   162 numbers from one linear layer, so adjacent timesteps have no structure — the
   structural cause of "noisy-looking" trajectories. We project the first differences
   onto the first 24 DCT-II modes and integrate from the anchor
   (`code/models/model.py:velocity_projector`). This is idempotent, rank 24, and
   reproduces constant-velocity motion **exactly**.
   * A *position*-space version was tried first and had to be abandoned: the DCT's
     even extension makes a constant-velocity ramp non-representable, giving a
     systematic `0.2744·(v·dt)` endpoint bias (0.41 m at 15 m/s) that the "99.9996 %
     energy retained" statistic completely hid. Switching to velocity space removed
     the bias (98.2 % reduction) and dropped the minFDE floor from 0.171 m to 0.
3. **Metric-weighted loss.** A per-element normalised MSE weights early timesteps
   ~2·10⁶ times more than late ones (`sd` spans 0.023 → 33 m), while minADE/minFDE
   average over timesteps in metres. Weighting the loss by `sd²/mean(sd²)` makes each
   timestep contribute equally in metres. This improved **all three arms on every
   metric** (−2…−4 % minADE, −4…−5 % minFDE, −0.03 miss rate).

---

## 5. Correctness verification (not "it runs")

* `code/models/selftest_overfit.py` — single-scene memorisation, judged **against the
  constraint's own floor**: loss drops 10³–10¹⁰×, sampled output lands on the floor
  (ADE/floor = 1.00×), roughness 0.19× ground truth. A deliberately under-trained
  control run correctly **fails**, so the criterion has teeth.
* `code/models/selftest_sampler.py` — an oracle field returning the true x₀ is driven
  through the real sampler: `max|err| = 0.000e+00` for steps ∈ {1, 2, 4, 8, 15, 50,
  256} for **both** paradigms. This separates "sampler is wrong" from "model is
  under-trained". It also implies discretisation error comes *only* from an imperfect
  x̂₀ — so "just add steps" cannot fix a bad field, which is exactly what the step
  sweep shows.
* `code/data/verify_cache.py`, `verify_precision.py` — shapes, contract, split
  disjointness, city coverage, and an independent float64 recomputation.
* `code/metrics/traj_metrics.py` self-test includes a regression guard that the
  ground truth's own kinematic violation rate is ≈0.05, not 1.000.

---

## 6. Known limitations (measured, not guessed)

| limitation | number |
|---|---|
| over-smoothing | roughness 0.07× ground truth (the constraint allows 0.37×, so the bottleneck is conditional-mean shrinkage, not the constraint) |
| candidates too narrow | K=6 lateral span median **2.2–2.65 m** vs ground-truth lateral p95 **19.7 m** |
| sharp turns at speed fail | turn coverage 0.26 (fm) / 0.37 (ddpm) in the ≥10 m/s bin; 0.43 / 0.50 in the 1–3 m/s bin |
| no map input | "pull out of a standstill and turn" has no kinematic evidence in 2 s of history |
| selector not yet implemented | the second half of the project (see §7) |

**Negative results** (kept because they are informative):

* **best-of-K (WTA) does not transfer to a sampler-integrated field.** WTA only sends
  gradients to the best of K noise draws, leaving the field undefined over most of
  noise space; the sampler queries all of it, so integrated quality collapsed
  (2.539 vs 1.757 minADE, +45 %).
* **Sampling temperature is a trade-off, not a free lunch.** T>1 widens candidates and
  raises turn coverage substantially but costs minADE on the full split; mixing
  temperatures across the K candidates gives no gain outside seed noise.

---

## 7. Next steps

1. **Selector** (the project's headline). `EXPERIMENTS_实验协议.md` §6 specifies it:
   label each scene with the cheapest model meeting a quality threshold τ on split 1,
   fit two cascaded binary gates (cost-ordered 1× < 4× < 15×), evaluate the resulting
   routing policy on split 2 against the three fixed models on an accuracy–latency
   Pareto front. A **go/no-go gate** comes first: if the oracle gap (best-per-scene
   minus best-single-model) is ≈0, a selector cannot help and we report that honestly.
2. **Map / lane geometry.** The root cause of the turn failures. nuPlan `.gpkg` maps
   are already extracted; needs rasterised lane features in the encoder.
3. **Mode-seeking objective that is compatible with sampling** (CFG-style guidance, or
   a repulsion term across the K candidates) — WTA is ruled out.
4. Multi-seed repeat of the full comparison (3 seeds done for the main table).

---

## 8. Repo layout

```
code/
  data/      extraction, cache building, normalisation, verification
  models/    shared encoder, DiT, unified model, self-tests
  metrics/   minADE/minFDE/miss/diversity/coverage-NLL/kinematics + calibration
  runners/   train, evaluate, visualise, and diagnostic scripts
results/     evaluation JSONs and figures referenced by the report
EXPERIMENTS_实验协议.md    authoritative experiment protocol (splits, fairness, metrics)
```

Diagnostic scripts are first-class citizens here — several of the bugs above were found
by a diagnostic, not by training. `code/runners/diag_*.py` covers constraint geometry,
direction bias, input ablation, turn coverage, sampling temperature and per-scene
numeric dissection.

---

## 9. Reproducing

```bash
# 1. build the cache from nuPlan mini .db files (needs nuplan-devkit data)
python code/data/build_mini.py --db-dir /path/to/mini --out /path/to/cache
python code/data/convert_cache.py --root /path/to/cache --force   # npz -> mmap npy
python code/data/make_norm.py     --root /path/to/cache
python code/data/verify_cache.py  --root /path/to/cache

# 2. verify the implementation before trusting any number
python code/models/selftest_overfit.py
python code/models/selftest_sampler.py

# 3. train one arm (reg | fm | ddpm) — identical config, only --objective differs
python code/runners/train.py --objective fm --steps 16000 --batch 64 --lr 3e-4 \
    --loss-power 2.0 --basis-m 24 --out runs/fm_lp2

# 4. evaluate all three on the held-out split
python code/runners/evaluate.py --split 2 --n 4096 --K 6 --seeds 3 --suffix _lp2

# 5. figures
python code/runners/viz_gallery.py --suffix _lp2 --out gallery.png
python code/runners/viz_stats.py   --suffix _lp2 --out stats.png
```

`requirements.txt` lists the Python dependencies. The reference environment is a single
consumer GPU; the largest model is 3.69 M parameters and training one arm takes
~15 min at 16,000 steps.

---

## 10. References

1. Zheng Y. et al. *Diffusion-Based Planning for Autonomous Driving with Flexible Guidance.* ICLR 2025.
2. Yang B. et al. *Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving.* CVPR 2024.
3. Tan T. et al. *Flow-Planner: Bridging the Gap Between Traffic Simulation and Real-World Deployment.* NeurIPS 2025.
4. Caesar H. et al. *nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles.* CVPRW 2021.
