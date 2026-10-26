# Second Project Report

**Which Model When: Comparing Three Ways to Predict Where a Car Will Drive**

Members: Member A, Member B, Member C, Member D · Code: https://github.com/`<username>`/`<repo>`

## 1. The data we use

We use **nuPlan mini**, a public self-driving dataset: 64 driving recordings from four
cities (Las Vegas, Singapore, Boston, Pittsburgh), at 10 frames per second.

We cut the recordings into short examples. A model sees the **last 2 seconds** of one
car's motion plus up to **10 nearby cars**, and predicts where that car will be over the
**next 8 seconds** (80 positions). Each example is rotated so the car starts at the
origin facing forward.

After removing cars that barely moved, **297,875 examples** remain. We split them by
**whole recording** — never by frame, so the same trip cannot appear in training and
testing — and kept all four cities in every part: 175,522 for training, 61,327 for
training the model selector, and 61,026 held out for final testing.

## 2. The three models we compare

* **A — Regression:** one direct guess. Very fast.
* **B — Diffusion:** refines random noise in 15 small steps into a trajectory; can give
  several plausible futures.
* **C — Flow matching:** a cheaper variant of the same idea, needing only **4** steps.

All three share the same input encoder and the same **3.69 M-parameter trajectory
network**; only the training rule and the generation procedure differ. Otherwise a win
could simply mean "the bigger network won". All three also obey one fixed physical rule:
the path must be smooth, realistic, and must start exactly at the car's current position.

## 3. What we have so far

Results on the **61,026 examples no model ever saw** (three random runs averaged; error
is the distance to the true position, averaged over 8 s, best of 6 candidates):

| model | network calls | average position error | error at 8 s | time per prediction |
|---|---|---|---|---|
| baseline: keep current speed | 1 | 5.94 m | 16.88 m | — |
| A regression | 1 | **2.82 m** | 8.25 m | **6 ms** |
| C flow matching | 24 | **1.83 m** | 4.83 m | 19 ms |
| B diffusion | 90 | **1.74 m** | 4.53 m | 60 ms |

1. Both generative models clearly beat regression (about **1.5×**); all three beat the
   "keep current speed" baseline by **3× or more**.
2. **Our second research question is answered: flow matching is nearly as good** — only
   **5 %** worse than diffusion at about **a quarter** of the computation. Giving it
   more than 4 steps changes nothing.
3. We verified the implementation rather than assuming it works: given a perfect target,
   generation reproduces it exactly (error 0.000), while a deliberately broken control
   correctly fails.

**Model selector (in progress).** The goal is to pick the *cheapest* model that is still
good enough. Perfect per-case selection would cut the error by 22 %, but that is
misleading: the minimum of several noisy measurements always looks better than reality.
Choosing the model on **half** of each model's candidates and scoring on the **other
half** leaves a real gain of only **4.7 %**. Our learned selector recovers **1–2 %** —
about a third of what is available.

## 4. Problems we ran into

**Data.** Positions are stored as huge global coordinates; converting them to local size
in low precision shifted every position by up to **25 cm**. Recorded positions also
**jitter between frames**, so even the ground truth looks physically impossible when
differentiated, and our plausibility check had to be made relative to the data.

**Models.** Our first version predicted the *noise* instead of the trajectory, which
amplified small errors into unusable output. Our first comparison was **unfair** (one
model had 8× fewer parameters) and was corrected. The standard trick for varied
predictions — training on the best of many samples — **fails badly here**, because it
leaves the generator undefined for most random inputs.

**Known weaknesses.** Predictions are too smooth, so turns are underestimated. The 6
candidate futures cover about **2 m** sideways where the truth spreads about **20 m**.
Sharp turns at high speed are usually missed, and there is no map information yet.

## 5. Next steps

1. Add road and lane geometry — the most likely fix for missed turns.
2. Improve the selector with better input features.
3. Replace the failed "best of many samples" idea with one that works for step-by-step
   generation.
4. Repeat with more random runs, broken down by scenario type.
