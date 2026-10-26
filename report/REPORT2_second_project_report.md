# Second Project Report

**Which Model When: Comparing Three Ways to Predict Where a Car Will Drive**

Members: Member A, Member B, Member C, Member D ·
Code: https://github.com/Parick100/nuplan-model-selector

## 1. The data we use

**nuPlan mini**, a public self-driving dataset: 64 driving recordings from four cities
(Las Vegas, Singapore, Boston, Pittsburgh) at 10 frames per second.

We cut the recordings into examples. A model sees the **last 2 seconds** of one car's
motion plus up to **10 nearby cars**, and predicts that car's position over the **next
8 seconds** (80 points). Each example is rotated so the car starts at the origin facing
forward.

After removing cars that barely moved, **297,875 examples** remain, split by **whole
recording** (never by frame, so the same trip cannot appear in both training and
testing) with all four cities in every part: 175,522 for training, 61,327 for training
the selector, 61,026 held out for testing.

## 2. The three models we compare

* **A — Regression:** one direct guess. Very fast.
* **B — Diffusion:** refines random noise in 15 small steps into a trajectory; gives
  several plausible futures.
* **C — Flow matching:** a cheaper variant needing only **4** steps.

All three share the same encoder and the same **3.69 M-parameter trajectory network**;
only the training rule and the generation procedure differ — otherwise a win could
simply mean "the bigger network won". All three also obey one fixed physical rule: the
path must be smooth, realistic and start exactly at the car's current position.

## 3. What we have so far

Evaluated on **4,096 randomly chosen scenes** from the 61,026 that no model ever saw
(three random runs averaged; error = distance to the true position, averaged over 8 s,
best of 6 candidates). This is **open-loop**: we score against the recorded future, not
a full driving simulation.

| model | network calls | average position error | error at 8 s | time per prediction |
|---|---|---|---|---|
| baseline: keep current speed | 1 | 5.94 m | 16.88 m | — |
| A regression | 1 | **2.82 m** | 8.25 m | **6 ms** |
| C flow matching | 24 | **1.83 m** | 4.83 m | 19 ms |
| B diffusion | 90 | **1.74 m** | 4.53 m | 60 ms |

1. Both generative models clearly beat regression (about **1.5×**), and all three beat
   the constant-speed baseline by **3× or more**.
2. **For the models we trained here, flow matching is nearly as good as diffusion** —
   5 % worse at about a quarter of the computation. That is a statement about *our*
   trained pair, not about the two methods in general; more than 4 steps does not help.
3. We verified the implementation rather than assuming: given a perfect target,
   generation reproduces it exactly (error 0.000), while a broken control fails.

**Model selector (in progress).** The goal is to pick the *cheapest* model that is still
good enough. The ceiling first: assigning every scene its best model perfectly would cut
the error from 1.72 m to 1.35 m (**21 %**). Our two trained selectors recover **5–12 %**
of that ceiling. The limit is not the ceiling but our ability to predict it: whether flow
matching or diffusion wins on a given scene is essentially **unpredictable** from scene
features (correlation −0.03) and is mostly sampling noise anyway, while
regression-versus-generative is a **real** per-scene difference (split-half reliability
0.60) that is still only weakly predictable (0.15). The lever is better input features,
i.e. map and lane geometry.

## 4. Problems we ran into

**Data.** Positions are stored as huge global coordinates; converting them to local size
in low precision shifted every position by up to **25 cm**. Recorded positions also
**jitter between frames**, so even the ground truth looks physically impossible once
differentiated, and our plausibility check had to be made relative to the data.

**Models.** Our first version predicted the *noise* instead of the trajectory, which
amplified small errors into unusable output. Our first comparison was **unfair** (one
model had 8× fewer parameters) and was corrected. The standard trick for varied
predictions — training on the best of many samples — **fails here**: it leaves the
generator undefined for most random inputs.

**Known weaknesses.** Predictions are too smooth, so turns are underestimated. The 6
candidates cover about **2 m** sideways where the truth spreads about **20 m**. Sharp
turns at high speed are missed, and there is no map information yet.

## 5. Next steps

1. Add road and lane geometry — the most likely fix for missed turns.
2. Improve the selector with those features.
3. Replace the failed "best of many samples" idea with one that suits step-by-step
   generation.
4. Repeat with more random runs, broken down by scenario type.
