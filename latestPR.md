## What & why

Fixes the two standing complaints — **reward freeze** and **convergence not stable** — plus the over-wide search box. Three planned fixes, and one extra fix that end-to-end testing proved was needed to avoid regressing the convergence gate.

### Planned fixes
- **Reward freeze (KNOWN_ISSUES #1)** — after the FAA window filled during warm-up, every later read pushed zero new samples, so `raw_faa`/`reward()` froze on the warm-up window and the optimizer hill-climbed one constant. `_pump()` now slides the window by ~one update interval of fresh samples per read in steady state (warm-up path unchanged).
- **Clamp bounds (KNOWN_ISSUES #6, geometry half)** — `optimizer.bounds` 3.0 → 1.5. PCA whitens per-axis, so z is in anchor-bank std units and the anchors sit at ~1 std; 3.0 let the 8-axis box corner reach ~8.5 std, deep in incoherent embedding space. 1.5 still reaches every anchor while pulling the corner in to ~4.2 std. (Verified this is **not** the convergence lever — bounds=3.0 with the new margin still fails.)
- **Relative RECOVER/SETTLE (convergence stability)** — FAA reward is a baseline z-score centered near 0. RECOVER now counts only rewards below a `-0.25` margin (was any `reward < 0`, which fired constantly on baseline noise); SETTLE keys off a low-variance plateau (`_recent_std`) instead of an instantaneous high reading (0.55 was unreachable for a clipped z-score; now 0.30 + plateau).

### The extra fix testing forced
`run_mock_optimizer.py` (the convergence gate) showed the conservative RECOVER margin **broke convergence** — the optimizer stalled in `explore` and never settled. Root cause: the old `reward < 0` RECOVER fired constantly and its `revert_to_best`+widen was secretly doing the real optimization work, escaping the cold start (`_current_reward_estimate=0.0` rejects every below-baseline move) and mid-run plateaus. Margin tuning can't satisfy both signals — anything aggressive enough for the mock's ~0 stall also thrashes on FAA's noisy-around-0 baseline.

Resolved by **decoupling convergence from absolute-reward RECOVER**:
- **Cold-start bootstrap** — seed `_current_reward_estimate` to `-inf` so the search accepts its first real move and adopts that reward as the baseline, instead of demanding every step beat an optimistic 0.0.
- **Stagnation escape** — when reward sits on a *low-variance plateau too low to SETTLE*, kick like RECOVER (revert + widen). Fires on a genuine stall but **never on FAA baseline noise** (high-variance → not a plateau). Proven: the FAA-mock state distribution is byte-identical with the escape on vs off, while the mock optimizer now settles on every seed.

Also restores a **pre-existing red test** (`test_hill_climb_reverses_on_clear_drop`, failing on a clean tree): the reversal did a full-magnitude negation while the test documents an intended `×0.75` damping. Restored the damping (near-neutral in the full loop since `set_step_size` renormalizes each step).

## Results
- `pytest -q`: **39 passed** (was 34 passed / 1 pre-existing failure).
- Fix A: `raw_faa` varies across consecutive `read_reward()` calls (the exact regression #1 reproduced) — new `tests/test_service.py`.
- Convergence: mock optimizer now **settles on every seed at 45–70% closer** (was 0–30% and stuck in `explore`), with the FAA-correct conservative margin retained.

## Test plan
- [x] `pytest -q` green (39 passed)
- [x] `python scripts/run_mock_optimizer.py --dry-run --seed {0,1,3,5,7}` → all reach `settle`
- [x] `python scripts/run_real_eeg_optimizer.py --mock --dry-run --baseline 5` → `raw` varies step to step; stagnation escape confirmed inert on the high-variance FAA signal
- [ ] **Needs a GPU box:** frame coherence across the `z` box at `bounds: 1.5`, and the `bounds` 1.2–2.0 sweep (the one remaining empirical knob). Restart any running diffusion server after the config edits (KNOWN_ISSUES #5).

## Note on the base
This branch is stacked on `aa033c5` (the anchor-bank restore for KNOWN_ISSUES #6), which is **not yet on `main`**. Rebasing onto `main` would reintroduce the `["cat"]` anchor-bank collapse, so the branch keeps that commit and this PR carries it too. It'll drop out of the diff automatically if the anchor-bank fix lands on `main` first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
