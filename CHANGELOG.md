# Changelog

## 2026-09-01 — CI reproduction for the bridge-ablation claim, and a real scale-threshold finding

**What was added:** `benchmarks/run_bridge_ablation.py` now takes `--ci` (fast smoke test on a
small committed fixture) and `--publish` (writes `ci_results/latest.json` +
`ci_results/badge-regularization.json`, the source for the README badge and for
jasonstiltner2026's build-time fetch). `benchmarks/generate_ci_fixture.py` generates the
committed fixture (`benchmarks/fixtures/mnist_ci_fixture.pt`) from real MNIST, so CI never
needs a live download for the fast path. `.github/workflows/ci-reproduce.yml` has two jobs
with deliberately different cadence — see below for why.

**What was found while building this:** the bridge-ablation accuracy effect (the "+89%"
claim) does not degrade gracefully to a small, fast CI budget the way the GCL repo's claims
do. Three fixture sizes were tried:

| Fixture | Result |
|---|---|
| 800 images/task, batch_size=64, 1 epoch | Noisy, direction-flipping — later traced to the CMS optimizer's slow tier (`slow_frequency=100` cumulative batches) never actually triggering at this batch count |
| 800 images/task, batch_size=8 (fixed the trigger issue), 3 epochs | Still noisy — 0-1 of 5 settings showed a bridge accuracy benefit |
| 6,000 images/task, batch_size=64, 3 epochs (10x bigger fixture) | **Stable, but consistently wrong direction** — bridges *hurt* accuracy in 5/5 settings, while *helping* forgetting in 4/5 |

That third result was the important one: going from noisy to stable-but-wrong-direction as
the fixture got bigger means this isn't "small n, same effect, more noise" (which is what
GCL's Punishment Paradox and Hart-Moore claims looked like at reduced scale) — it's a real
threshold effect. The bridge accuracy benefit appears to need something close to the full
~60k-image MNIST (real per-task, ~12k images/task on the real 5-way Split-MNIST) before it
shows up.

Confirmed directly: a full-scale run of this exact script (real MNIST, same 5 regularization
strengths as the site's chart, `benchmarks/run_bridge_ablation.py --publish`, ~11 minutes)
reproduces the historically committed `experiments/results/pareto_analysis.json` values to
3 decimal places, and gives **+88.5%** at reg=5.0 (bridges improve accuracy in 5/5 settings)
— matching the published "+89%" claim. That data was already solid; this run is the first
time it's been re-derived from a live CI/notebook path rather than existing only as a
committed result file.

**Why two CI jobs, not one:**

- `smoke-test` (every push to `main`, ~2 min, fixture-based) — proves the code path still
  runs after a change. Does **not** publish `ci_results/` — a small-fixture run of this
  particular experiment is not a reliable stand-in for the real claim (see table above), so
  publishing it as "the" CI-verified number would be misleading in the other direction from
  the fabrication problem the sibling GCL repo had.
- `reproduce-full` (weekly + `workflow_dispatch`, real MNIST, ~10-12 min) — this is what
  actually publishes `ci_results/latest.json` and the badge. Slower and less frequent than
  the GCL repo's every-push cadence, stated plainly in the README rather than glossed over.

**Also fixed (jasonstiltner2026 repo, same day):** `RegularizationChart.tsx`'s raw
`withBridges` display values at reg=1.0/5.0/10.0/20.0 had small copy-paste drift from this
data (e.g. `0.189` shown for reg=5.0 vs. the real `0.185`) — the improvement percentages were
already correct throughout, only the raw accuracy numbers needed correcting. Also corrected
"CIFAR-scale validation" to "Split-MNIST validation" in three places — the actual backing
script trains on Split-MNIST, not CIFAR.
