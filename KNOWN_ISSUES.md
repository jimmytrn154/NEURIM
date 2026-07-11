# Known issues / open work — handoff summary

Status snapshot of unresolved problems found while wiring up the StreamDiffusion
real-time path and the new frontend-app dashboard. Ordered by severity/urgency.

## 1. [CRITICAL, unfixed] FAA reward freezes solid a few seconds into every real-EEG session

**Where:** `src/signal_service/service.py`, `FAARewardSource._pump_until_ready()` (lines 30-35).

```python
def _pump_until_ready(self) -> None:
    if self._stream is None:
        self._stream = self.eeg_source.stream()
    while not self.computer.ready():
        _t, sample = next(self._stream)
        self.computer.push_sample(sample)
```

`FAARewardComputer`'s internal buffers are proper sliding windows (`deque(maxlen=N)` -
see `src/signal_service/faa.py:110-125`), correctly designed to keep advancing as new
samples arrive. But this pump function only pushes samples **until `ready()` first
becomes `True`**. Every subsequent call re-enters the `while` loop, finds `ready()`
already `True`, and pushes **zero** new samples. The window's content is then frozen
forever - `raw_value()`/`reward()` return the exact same number for the rest of the
session.

**Confirmed empirically**, not theoretical: ran `scripts/run_real_eeg_optimizer.py --mock
--dry-run --baseline 2` and got identical `reward`/`raw` values across 100 consecutive
optimizer steps (see conversation log). This affects every real-EEG code path:
`run_demo.py` (real-EEG mode), `run_real_eeg_optimizer.py`, and presumably
`run_streamdiffusion_demo.py`, since all three go through `build_faa_service` /
`FAARewardSource`.

**Practical impact:** in a real session, FAA reward would look real for the first
`window_s` (2s), then go completely static - the optimizer would hill-climb against
a single frozen number for the rest of the session. This is a strong candidate
contributor to the earlier "convergence not stable" complaints, independent of the
anchor-prompt issues (below).

**Proposed fix** (not yet applied - pending confirmation):
```python
def _pump_until_ready(self) -> None:
    if self._stream is None:
        self._stream = self.eeg_source.stream()
    if not self.computer.ready():
        while not self.computer.ready():
            _t, sample = next(self._stream)
            self.computer.push_sample(sample)
    else:
        _t, sample = next(self._stream)  # keep the window sliding
        self.computer.push_sample(sample)
```
Open question: one sample per call is the minimal fix, but the "correct" number to
pull per call (to keep the window's real-time alignment tight against
`update_interval_s` cadence) may need more thought - worth a second pair of eyes.

---

## 2. [Design gap] No script bridges real/mock EEG to the WebSocketOrchestrator hub

**Where:** `src/orchestrator/orchestrator.py`'s `WebSocketOrchestrator` expects an
external client to dial in with `{"role": "signal"}` and stream `RewardMessage` JSON
(`_handle_signal_client`, lines 143-145). No such client exists anywhere in the repo.

- `scripts/run_demo.py` has two mutually exclusive branches: plain mode processes real
  EEG but never touches the hub (fully in-process `LocalOrchestrator`); `--serve` mode
  starts the hub but has **no** `eeg_source`/`signal_service` at all (lines 85-91 of
  `run_demo.py`).
- `scripts/test_faa_stream.py` reads real/mock EEG but only `print()`s to console - no
  networking of any kind.

**Why it matters now:** the new `frontend-app` (Next.js dashboard,
`components/neurim-dashboard.tsx`) connects to `ws://localhost:8765` as a **display**
client and expects the hub to be reward-driven. Without a signal-client bridge, the
hub will only ever show one static initial frame - no morph, no responsiveness, since
`observe_reward()` is never called.

**Not yet built.** Needs a new script: real/mock EEG → FAA reward → dial
`ws://<hub>:8765` with `{"role": "signal"}` → stream `RewardMessage`s in the wire
format `WebSocketOrchestrator` expects.

---

## 3. [RESOLVED] `/anchors` endpoint gap - and the server it applied to no longer exists

Originally: `run_diffusion_server.py` (SDXL-Turbo) had a live `/anchors` endpoint;
`run_streamdiffusion_server.py` (the StreamDiffusion-based backend) didn't, so the
frontend-app's "Apply anchors" button and `--set-anchors` in the optimizer scripts
would 404 against it.

**Since resolved from two directions:**
1. `run_streamdiffusion_server.py` gained its own `/anchors` route
   (`LatentWalkRenderServer.set_anchor_prompts()`), matching the SDXL server's behavior.
2. More fundamentally, `run_streamdiffusion_server.py` **no longer uses StreamDiffusion
   at all** - see item 7 below, which replaces this whole file's rendering engine.
   `/anchors` still works the same way (re-fits `PCAProjector` against the new prompts,
   thread-safe under `render_server.lock`), just against a plain diffusers pipeline now.

---

## 4. [Security - user-side action needed, status unconfirmed] Leaked OpenAI API key

**Where:** `frontend-app/.env.example`, committed in `a25550a "feat: add .env.example
with OPENAI_API_KEY placeholder"`. The committed value is a real-looking `sk-proj-...`
key, not a placeholder, despite the commit message.

**Action required (flagged to the repo owner already, status unknown as of this
summary):**
1. Rotate/revoke the key in the OpenAI dashboard - treat as compromised regardless of
   whether the commit was pushed to the public GitHub remote.
2. Replace the value in `.env.example` with an actual placeholder.
3. Confirm the real key only ever lives in `frontend-app/.env` (already gitignored).

---

## 5. [Recurring operational gotcha, now documented] Config is read once at process startup

Both diffusion servers (and `run_streamdiffusion_demo.py`) call `Config.load()` once in
`main()` and fit the anchor-prompt projector immediately after - there is no
hot-reload anywhere in this codebase. Editing `config.yaml` (anchor prompts, `bounds`,
etc.) does **nothing** to an already-running process. This caused real confusion
earlier ("changed anchors to cat, still generating dogs") - root-caused to the server
process not having been restarted (and, separately, to local edits not having been
synced to the server's own checkout). Worth this being common knowledge for anyone
touching `config.yaml`.

---

## 6. [Resolved, but fragile] Anchor-prompt bank design is the dominant lever for search-space quality

Established but worth restating for whoever touches anchor prompts next (including
whatever the frontend's OpenAI-based anchor generator produces): `PCAProjector` yields
at most `n_anchors - 1` genuine directions, and those directions are only as diverse as
the *content differences* between anchors. A single anchor collapses the whole search
space to one point (this actually happened - anchors were briefly set to just `"cat"`,
described in item 3's history). Anchors varying only one attribute (e.g., fur color)
mean every reachable direction is a shade of that one attribute, even with many dims
nominally configured. Current cat/puppy banks (10 anchors each, `search_dims: 8`) were
manually authored to vary genuinely independent axes (breed/color/coat/pose) with a
fixed photographic scaffold. **Open question:** does the frontend's OpenAI-generated
anchor set (`app/api/anchor-prompts/generate/route.ts`) actually follow this same
discipline (parallel structure, independent axes, right count for `search_dims`)? Not
yet reviewed/validated.

---

## 7. [RESOLVED - architecture change] `run_streamdiffusion_server.py` no longer uses StreamDiffusion

The old `t_index_list` tuning saga (`[0]` broke img2img continuity - every frame
regenerated from near-scratch as an unrelated composition; `[32,45]`, StreamDiffusion's
img2img tutorial default, produced mosaic garbage on from-scratch bootstrap; `[0,16]`
was the empirically-found fix) is now moot, because the whole img2img-continuity
mechanism it was tuning is gone.

**What changed:** `scripts/continuous_dog_latent_morph.py` demonstrated a simpler,
officially-supported technique - pass an explicit `latents=` tensor (generated once,
seeded, fixed for the whole session) and `prompt_embeds=` directly to a plain
`diffusers.StableDiffusionPipeline`, rendering fresh from scratch every frame. No
img2img feedback loop, so no "how much of the previous frame should survive" question
to tune at all. `run_streamdiffusion_server.py` was rewritten around this - it no
longer depends on the StreamDiffusion package, the `utils/wrapper.py` sys.path hack,
the undocumented `stream.prompt_embeds` attribute override, or the separate
`streamdiffusion-env` (plain diffusers has the same dependency shape as
`run_diffusion_server.py`'s `torch-env`).

**Names that changed** (in case old references linger anywhere): `build_wrapper` →
`load_pipeline` (now returns `(pipe, fixed_latent, device, dtype)`, not a
`StreamDiffusionWrapper`), `StreamDiffusionRenderServer` → `LatentWalkRenderServer`,
`_fit_projector` gained a `device` parameter. `scripts/test_streamdiffusion.py` was
updated to match; anything else referencing the old names (docs, notes, muscle memory)
should be treated as stale.

**Still worth re-validating after this change:** run `test_streamdiffusion.py` again
with the new `--steps`/`--guidance-scale` flags before trusting it live - the technique
is proven by the reference script, but hasn't yet been executed against our actual
anchor-prompt bank on real hardware as of this writing.

---

## Suggested priority order for a fresh pair of eyes

1. **Item 1** (reward freeze) - blocks any real-EEG session from working correctly at
   all; highest impact, fix is drafted and small.
2. **Item 2** (missing signal-client) - blocks the new frontend-app from being reward-
   driven at all.
3. **Item 3** (`/anchors` on StreamDiffusion server) - blocks dynamic anchor updates
   on the currently-preferred real-time backend.
4. **Item 4** (leaked key) - not code, but time-sensitive; confirm it's actually been
   rotated.
