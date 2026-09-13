# NOTES.md — running log of decisions, deferrals, and open questions

Scratch file. `PROJECT.md` is the durable context; this is where things go that
do not belong there yet.

## Deferred to later phases (noted, not implemented)

Nothing yet. Ideas that belong to phases 2–4 get parked here rather than
sneaking into phase-1 code.

## Phase 1, step 1 — scaffolding

Decisions taken without asking, because they are implementation details rather
than modelling choices:

- **`GeneratorConfig.alpha_analytic`.** Each generator config exposes the
  analytic MSD exponent of `PROJECT.md` section 4 as a property, so validation
  tests read the target off the config instead of restating it. For
  `LevyFlightConfig` the property *raises*, and the config instead exposes
  `msd_is_finite = False` and `nu_analytic = 1/s` for the fractional moment.
  This makes the diverging-second-moment rule structural rather than a comment.
- **Cache key.** SHA-256 over canonical JSON (sorted keys, tight separators),
  truncated to 16 hex characters, prefixed by the config tag:
  `fbm-3f2a91c0d4e5b678.npz`. The prefix is only so a cache directory can be
  skimmed by eye.
- **Config stored twice.** Inside the `.npz` under `__config__` so an archive is
  self-describing if moved, and as a sibling `<hash>.json` so it is readable
  without numpy. `load_arrays` compares the stored config against the requested
  one and raises on mismatch rather than returning the wrong trajectories.
- **Atomic writes.** `save_arrays` writes to a `.tmp<pid>` file and renames, so
  an interrupted run cannot leave a truncated archive that a later run loads.
- **`FitConfig`.** The `[lag_min, T/10]` fit window of `PROJECT.md` section 2 is
  a config, not a set of default arguments scattered over the estimators.
- **Editable install.** `pip install -e . --no-deps` so notebooks import the
  package without `sys.path` surgery; `pythonpath = ["src"]` in
  `pyproject.toml` keeps pytest working without the install.

Parameter *semantics* for CTRW, Lévy walk and diffusing diffusivity are written
into the config docstrings now but only become load-bearing in step 2; if a
generator needs a parameter the config does not carry, the config changes first.

## Phase 1, step 2 — generators

### Findings that affect later phases

- **The fit standard error is not the uncertainty on the exponent.** The ensemble
  MSD is built from the same particles at every `t`, so its points are strongly
  autocorrelated and the ordinary-least-squares error is far too optimistic:
  measured over twelve seeds, the reported error is ~0.0005 while the actual
  seed-to-seed standard deviation of the fitted exponent is ~0.012 (N = 2000,
  T = 1024) — a factor of ~25. Phase 2 is an estimator-variance study, so it
  must get its error bars from resampling over particles, never from
  `PowerLawFit.exponent_err`. That field is a goodness-of-fit diagnostic only,
  and is documented as such.

- **CTRW finite-time bias changes sign with `a`.** Expanding the Laplace
  transform of the pure Pareto density gives
  `<n(t)> ≈ (t/τ0)^a / (Γ(1-a)Γ(1+a)) · [1 + a/((1-a)Γ(1-a)) (τ0/t)^(1-a)] - 1`.
  The relative correction decays only as `(t/τ0)^-(1-a)` and dominates as
  `a → 1`; the additive `-1` dominates when `a` is small, because `<n>` is then
  only of order ten events. The two pull opposite ways, so the fitted-exponent
  bias at T = 4096 runs `+0.069` at `a = 0.3`, `+0.016` at `0.5`, `-0.015` at
  `0.7`, `-0.071` at `0.9`, crossing zero near `a ≈ 0.55`. Verified against the
  measured renewal count (predicted ratio to the leading term 0.906 vs measured
  0.910 at `a = 0.3`; 1.067 vs 1.074 at `a = 0.7`). Consequence: **a naive
  "the bias shrinks as T grows" test would be wrong**, and phase 2 sweeps over
  `a` must not read the residual bias as an estimator property — it is a
  property of the observation window.

- **The Lévy-walk MSD estimator gets noisier with time.** Since
  `<r^4> ~ t^(5-g)` outgrows `<r^2>^2 ~ t^(6-2g)`, the relative standard error
  of the ensemble MSD grows roughly as `t^((g-1)/2)`: measured at `g = 1.4`,
  N = 3000, it runs from 1.9 % at `t = 10` to 5.7 % at `t = 4095`. This is a
  second, independent reason to cap the fit window that has nothing to do with
  the window-count argument behind `[lag_min, T/10]`. A variance-weighted fit
  would likely help and is worth trying in step 3 — noted, not implemented.

- **Davies–Harte never failed.** Over `H` in `[0.02, 0.98]` × `n` in
  `{8, 16, 64, 257, 1024}` the smallest circulant eigenvalue was `+2.3e-5`
  relative to the largest, so the `m = 2n` embedding is valid throughout the
  usable range. The non-negativity guard is therefore unreachable from a valid
  config and is unit-tested directly on a poisoned eigenvalue array instead.

### Decisions taken without asking

- **Two resamplers, not one.** `resample_step` (zero-order hold) for the CTRW and
  `resample_linear` for the Lévy walk. The difference is the physics: using the
  step rule for a Lévy walk would destroy the space-time coupling that makes its
  second moment finite. Both are tested against hand-built event sequences.
- **Both resamplers refuse to extrapolate.** If any particle's event sequence
  ends before the last grid point they raise, because holding the last known
  position would fake a long trapping event and bias the MSD down. Event
  generation therefore draws in doubling rounds until every particle is covered.
- **fGn two-for-one.** The Davies–Harte transform is a proper complex Gaussian
  with real covariance, so its real and imaginary parts are two independent fGn
  samples. One FFT yields two series; a test asserts their cross-correlation is
  below 0.02.
- **SBM and DDM integrate their exact transition laws**, not Euler steps —
  increment variance `2 D0 (t_{i+1}^α - t_i^α)` for SBM, and the exact OU update
  for DDM. Both matter where phase-2 sweeps will push the parameters (small `α`,
  `τ → dt`).
- **Lévy flight in 2-D uses independent components**, giving non-circular
  contours. The isotropic variant needs a sub-Gaussian construction. Nothing in
  phase 1 uses an angular observable, but this must be revisited before any
  2-D Lévy-flight result is quoted.
- **The Lévy walk is the ordinary, non-equilibrated one** — flights start fresh
  at `t = 0` rather than being aged. This shifts the amplitude and the approach
  to the asymptotic exponent, not the exponent itself.
- **`ea_msd` + `fit_powerlaw` were written in step 2, not step 3**, because a
  generator cannot be validated without them. The alternative was a second
  fitting implementation inside `validation/`, which would have been worse. The
  rest of `PROJECT.md` section 5 is untouched and still belongs to step 3.
- **Two fit windows are distinguished, deliberately.** `[lag_min, T/10]` applies
  to TA-MSD versus *lag*, where the number of contributing windows collapses.
  For EA-MSD versus *time* that argument does not apply — every particle
  contributes at every `t` — so the window is instead a statement about where
  the asymptotics hold, and every caller states it. `fit_powerlaw` has no
  default window.

## Step-2 review corrections

- **The `a = 0.3` bias discrepancy was mine, not a convention difference.** The
  reviewer's `+0.046` is right; my `+0.069` contained an extra `+0.023` of
  **1/N Jensen bias** from fitting `log(MSD̂)` at `N = 2000`. Because
  `E[log MSD̂] ≈ log MSD − σ²_rel/2` and the CTRW's relative variance *falls*
  with `t` at small `a` (many particles have not jumped at all early on), the
  log-space fit tilts upward. Measured MSD-fit bias at `a = 0.3`, T = 4096:
  `+0.067` (N = 2000), `+0.052` (N = 8000), `+0.042` (N = 20000), converging on
  the renewal-count value `+0.045` and the closed-form `+0.043`. Not `tau0`
  (that would need `tau0 = 4` to reproduce `+0.069`) and not the counting
  convention. **This bias affects any log-log fit whose relative variance varies
  with the abscissa**, so phase 2 must watch for it wherever ensembles are
  small; the bootstrap gives a correct interval but does *not* remove it.
- **Sign change at `a = 0.604`, not 0.55.** My 0.55 came from interpolating the
  N-contaminated measurements. The closed form gives 0.604 at `T = 4096`,
  drifting to 0.566 by `T = 65536` — so the crossing is itself T-dependent.
- **`N^(-1/2)` scaling holds.** `SD·√N` is flat within ±10 % over
  `N ∈ {500, 1000, 2000, 4000, 8000}` for Brownian, fBm and CTRW — and ±10 % is
  the expected scatter of an SD estimated from 24 seeds. One correction to the
  review's premise: the `0.012` I reported was at `N = 2000`, not `N = 4000`;
  at `N = 1000` the measured value is `0.0169`, not `0.024`.

## Phase 1, step 3 — estimator toolbox

- **EB plateau converges slowly from above.** CTRW `a = 0.5`, lag 2, N = 1500:
  `0.705` (T = 1024), `0.627` (4096), `0.580` (16384) against the theoretical
  `π/2 − 1 = 0.5708`. Tests use `T = 16384`. EB also rises with `lag/T`, so the
  plateau must be read at small lag: at `T = 65536` it runs 0.604 (lag 2) to
  0.843 (lag 512).
- **`nu(q)`: the high-`q` branch is accurate, the low-`q` branch is biased up
  ~0.02 at any affordable `T`.** The ballistic cone contributes `t^(q+1-g)` to
  *every* moment, and for `q < g` that decays relative to the `t^(q/g)` bulk
  only as `~t^(-1/3)` at `g = 1.5`. Measured 0.689 vs theory 0.667 even at
  `T = 16384`. This is why the kink is located by fitting the one-parameter
  analytic family rather than intersecting two regression lines — the fitted `g`
  then recovers the truth to 0.001–0.025 across `g = 1.4/1.5/1.7`.
- **Peak memory is dominated by the generators, not the estimators.** At
  N = 4000, T = 4096 (one float64 array = 125 MiB): Lévy walk peaks at 1324 MiB
  and fBm at 937 MiB — the event arrays and the complex FFT buffers
  respectively, roughly 10× and 7.5× the output. Phase 2 sweeps should generate
  in particle chunks rather than raising `n_particles`.
- `ta_msd` and the EB curve cost ~1.9 s per 32 lags at this size, linear in the
  lag count. `moment_spectrum` at 16 `q` values costs ~3.7 s, essentially 16
  independent fits.

## Phase 1, steps 4–5 — visualisation and notebooks

- **Palette validated, not eyeballed.** Seven categorical slots, worst adjacent
  separation `ΔE = 9.1` under protanopia and `19.6` for normal vision (floors 8
  and 15). Three slots (aqua, yellow, magenta) sit below 3:1 contrast on the
  light surface, so a legend is mandatory on every multi-series figure and the
  curve plots direct-label as well. Colour follows the *mechanism*; two Hurst
  values of fBm share a hue and differ by line style.
- **Direct labels only for ≤ 4 series.** With seven they collide; the legend
  moves outside the axes instead. Overlapping fit windows are merged into
  disjoint spans before shading, or the alpha stacks and invents a darker band.
- **`plt.subplots` retains figures.** Animations need a pyplot canvas, so the
  figure functions use it, which means a loop over mechanisms must
  `plt.close(fig)`. Documented in `viz/__init__.py`; the test suite closes
  figures in an autouse fixture.
- **Inline animation size scales as `frames × dpi²`.** `display_animation`
  base64-encodes every frame. Default dpi dropped to 90 and `max_frames` to
  200/180/150 for single/few/cloud; the notebooks override to 70–90 frames. Even
  so the two notebooks are 8.4 MB and 9.7 MB with outputs.
- **`studies.py` is a new module** not in `PROJECT.md` section 3. The
  estimator-variance study is several dozen lines of generation and fitting, and
  "package, not notebook" leaves it nowhere else to live.

### The phase-1 result, in numbers

Single-trajectory TA-MSD exponent, 200 trajectories, `T` from 128 to 8192:

| mechanism | true α | median | IQR |
|---|---|---|---|
| Brownian | 1.00 | 0.96 → 0.99 | 0.215 → 0.070 |
| fBm H=0.3 | 0.60 | 0.60 → 0.59 | 0.182 → 0.062 |
| scaled Brownian α=0.6 | 0.60 | **0.93 → 0.98** | 0.215 → 0.070 |
| CTRW a=0.5 | 0.50 | **0.93 → 0.88** | 0.408 → **0.271** |

Two *different* failure modes, and the distinction matters for phase 2:

- **Scaled Brownian motion converges as tightly as Brownian, onto the wrong
  number.** Its IQR falls like `1/√T` while its median walks towards 1.0 against
  a true 0.6. A single-agent error bar will not warn you — it is precise and
  inaccurate. This is a stronger statement than "non-ergodic mechanisms are
  noisy" and it is now asserted in `tests/test_viz.py`.
- **CTRW fails both ways**: wrong median *and* an IQR that falls only 1.5-fold
  over a 64-fold increase in `T`. The residual width is the EB plateau.

Both CTRW and SBM report `α ≈ 1` from one trajectory because both have a TA-MSD
linear in the lag — for different reasons (weak ergodicity breaking vs ageing).
That coincidence is what makes single-agent mechanism classification hard, and
is the concrete form of the Q1 wall.

## PHASE1_REVIEW.md corrections

- **Item 1 (framing) — rewritten, and it was a real correction.** The old text
  said the CTRW spread "does not shrink." Measured: fitting `IQR ~ T^-p` gives
  `p ≈ 0.107` for CTRW against `p ≈ 0.26–0.28` for Brownian, fBm and scaled
  Brownian motion. The spread **does** shrink, about three times slower — the
  actual, stronger result is that CTRW and scaled Brownian motion *converge*,
  onto `alpha ≈ 1` against truths of 0.5 and 0.6. `studies.iqr_scaling_exponent`
  makes this a printed number instead of an eyeballed panel. `PHASE2_PROMPT.md`
  independently quotes these same `p` values, confirming the fix.
- **Item 2 (notebook disagreement) — resolved, not a bug.** `01`'s single
  sample (0.99) and `02`'s ensemble median (0.88) came from the *same* fitting
  method on the *same* mechanism. Two compounding, unrelated effects: (a) one
  sample from a distribution with IQR ≈ 0.25 landing high is unremarkable; (b) a
  CTRW at `a = 0.5` makes very few jumps — median 46 in `T = 8192`, 1st
  percentile just 1 — so the default `[1, T/10]` window's upper edge is
  supported by a handful of events and the single-trajectory TA-MSD has
  measurably saturated by then (median fitted slope 0.965 over lags [1,32],
  0.890 over [1,819], 0.787 over [100,819]). `studies.lag_window_sensitivity`
  makes this reusable; regression tests pin both effects down in
  `tests/test_phase1_review.py`.
- **Item 3 (NaN censoring) — was already reported, now made a first-class
  field.** `exponent_spread_study`'s `summary[label]["censored"]` gives the
  per-length failure fraction (CTRW `a=0.5`: 7.5%, 3.5%, 1.5%, 1.5% at
  `T = 128, 512, 2048, 8192`; zero for every ergodic mechanism). The `nan*`
  reductions were never silently dropping this information — the review's
  concern was that it wasn't *surfaced* — so nothing about the estimator
  changed; the summary dict now carries it explicitly.
- **Item 4 (missing quantitative checks) — added.**
  `estimators.ergodicity.brownian_eb` (`EB(Δ) ≈ 4Δ/(3T)`) is the positive
  control the CTRW plateau alone can't be: a plateau scaled by the wrong
  constant is still a plateau, but this formula has a specific slope *and*
  coefficient. Measured within 9% at every lag tested. `studies.ergodicity_checks`
  and `studies.benchmark_table` bundle both into printed, JSON-summarised
  results and are now called from `02_ensemble.ipynb`.
- **Item 5 (repo size / test organisation):**
  - Both notebooks now write animations to `results/media/*.mp4` and embed a
    `<video src="...">` referencing them, instead of `display_animation`'s
    inline base64 JSHTML. **19 MB → ~920 KB** combined. `results/media/` is
    gitignored; `results/*.json` is not (small, meant to be read).
  - `pytest` markers: `slow` registered in `pyproject.toml`, default
    `addopts = "-q -m 'not slow'"`. The three files that validate generators or
    estimators against real Monte Carlo ensembles
    (`test_generators_validation.py`, `test_estimators.py`,
    `test_step2_corrections.py`) are marked slow at module level; six
    individually slow tests elsewhere (`test_viz.py`, `test_phase1_review.py`)
    are marked per-test. Bare `pytest`: 125 tests, **2.47 s**. `pytest -m slow`:
    194 tests, 91.4 s. Combined: 319 — nothing lost in the split.
  - Title bug fixed (`"3000 superdiffusive walkers"` → `"1500..."`, matching
    `n_particles=1500`).
  - `nbstripout` added as a dev dependency; **not yet installed as a git
    filter**, because no git repository exists in this project yet — there is
    no history to rewrite (`git filter-repo` is moot). Run
    `pip install -e .[dev] && nbstripout --install` once the repo is
    initialised, before the first commit.
  - **Deferred, not done:** "consolidate by parametrisation... move analytic
    reference values into a single reference table module... roughly a third of
    the lines." This is a real refactor (touching all four generator/estimator
    test files) rather than a quick fix, and risks silently dropping assertions
    under time pressure. Flagging explicitly rather than skipping silently.

## Open questions for the next steps

- **CTRW jump kernel — still unanswered.** Implemented as Gaussian, so CTRW is a
  pure waiting-time mechanism with no Lévy statistics mixed in. This is the
  standard choice and the one Q2 needs (it keeps CTRW separable from the Lévy
  mechanisms), but it was not confirmed. Changing it later means re-validating.
- Diffusing diffusivity is parameterised the Chechkin way, `D = |Y|^2` with `Y`
  an `n_aux`-component OU process. `n_aux = 1` maximises non-Gaussianity.
- The repository is not yet a git repo (`.gitignore` is in place for when it is).
- Variance-weighted power-law fitting (see the Lévy-walk note above) — worth
  trying in step 3, would tighten every heavy-tailed exponent estimate.
- Isotropic 2-D Lévy flight via a sub-Gaussian construction, if any 2-D
  Lévy-flight result is ever quoted.

## Phase 2 — Identifiability and Single-Trajectory Regime Inference

### Core Findings & Numbers

1. **Feature Scale Invariance:**
   - 19 scale-invariant features grouped into 7 families (`tamds`, `acf`, `ngp`, `immobility`, `p_variation`, `excursion`, `nonstationarity`).
   - Every single feature passes strict scaling invariance $x \mapsto c \cdot x$ across all 6 generative mechanisms within numerical tolerance.

2. **Classification across $T$ & Pairwise Separation:**
   - **fBm vs CTRW:** Separates completely with 0.0 % error even at $T = 128$. Ablation confirms that `immobile_fraction` alone achieves 100 % accuracy.
   - **SBM vs Brownian:** Separates asymptotically as $T$ grows; pairwise error drops from 29.6 % ($T = 128$) to 5.0 % ($T = 8192$) driven by `increment_var_ratio` and non-stationarity.
   - **DDM vs Brownian:** Separated at short times via non-Gaussian parameter $a_2(\Delta)$, but approaches an information wall as observation window expands far beyond correlation time $\tau$.

3. **Information Floor (Whittle MLE vs Bound):**
   - For fBm ($H = 0.3$), Whittle MLE efficiency $\text{CRLB} / \text{Var}(\hat{H}) \to 0.97$ as $T \to 8192$, confirming asymptotic efficiency.
   - The Phase 1 naive TA-MSD estimator is wasteful: its efficiency falls to $0.094$ at $T = 8192$, showing that its spread is an estimator limitation rather than an absence of information in the data.

4. **Closing the Loop (Conditional Exponent Recovery):**
   - Overall Exponent MAE ($T = 2048$):
     - Naive (Phase 1 TA-MSD): **0.1709** (CTRW MAE: 0.3691, SBM MAE: 0.3487).
     - Predicted (Phase 2 Identification + Correction): **0.0567** (CTRW MAE: 0.0972, SBM MAE: 0.0813).
     - Oracle (Exact Mechanism Known): **0.0511**.
   - Gain from Phase 2: **66.8 % error reduction** ($0.1709 \to 0.0567$).
   - Misclassification cost: **0.0056** ($0.0567 - 0.0511$).

### Step 6 — Lévy walk exponent correction (2026-09-06)

**Trigger.** Review of Step 6 flagged `levy_walk` as the only mechanism where
`mae_naive == mae_oracle == mae_predicted` bit-for-bit in
`step6_conditional_exponent.json` — `estimate_exponent_conditional` fell
through to the naive Phase 1 TA-MSD fit regardless of whether the mechanism
was known or predicted. `PHASE2_GLS_PATCH.md` asked for the GLS-weighted fit
`PROJECT.md`'s fitting convention promises, wired in as the correction.

**Attempt 1 — GLS-weighted TA-MSD fit.** The prescribed fix: bootstrap the
lag-lag covariance of `log(TA-MSD)` and use it to weight the log-log fit.
Implemented and tested four ways: full covariance with ridge regularisation,
diagonal-only inverse-variance (reusing `fit_powerlaw`'s existing `use_gls`
path with per-time-block contributions swapped in for per-particle ones,
since a single trajectory has no ensemble to resample), rank-truncated
pseudo-inverse (keeping only the top-`k` covariance eigenmodes), and — as a
control to rule out bootstrap-estimation error — the *true* ensemble
covariance computed from 800 independently generated trajectories at fixed
`gamma`. Every variant came out worse than plain OLS; the true-covariance
control was dramatically worse (MAE ≈0.43 vs naive's ≈0.13 on the same
held-out set). Diagnosis: for a single trajectory the short lags are precise
(many overlapping windows) but biased toward the within-flight ballistic
regime, while the long lags are asymptotically correct but, for one
realisation, individually very noisy (a single trajectory's high-lag TA-MSD
point can be off by a factor of ~2). Inverse-variance weighting does exactly
what it is designed to do and concentrates weight on the precise-but-biased
end, which makes the bias worse, not the estimate better. Confirmed with the
true covariance, so this is not a bootstrap-implementation problem — it is a
mismatch between the weighting objective (minimise variance) and the actual
failure mode (lag-dependent bias).

**Attempt 2 — moment-spectrum kink fit.** `PROJECT.md` section 5 describes the
moment spectrum `nu(q)` as the Lévy walk's actual fingerprint — piecewise
linear with a kink at `q = g`, where fBm, SBM and CTRW are all flat — so
`fit_bilinear_spectrum` looked like the mechanism-correct tool, not a
reweighted derived statistic. It fails on one trajectory anyway:
`moment_spectrum` needs an ensemble to resolve high-`q` moments (its own
docstring: "high moments... need many more particles"), and a lone trajectory
has none. Direct single-trajectory fit: MAE ≈0.41 (much worse than naive).
Tried building a pseudo-ensemble instead, splitting the one trajectory into
16–192 non-overlapping segments and treating each as an independent particle
for the spectrum fit; swept segment count and the fit window/`q`-range. Best
tuned configuration still landed at MAE ≈0.147–0.15 on the held-out set naive
scores ≈0.13 on.

**Attempt 3 — flight-duration MLE (adopted).** Both prior attempts reweighted
or refit a statistic *derived* from the trajectory (TA-MSD, moment spectrum).
The CTRW correction that already works (`estimate_exponent_ctrw`) instead
does MLE directly on the process's own randomness — waiting-time durations.
The Lévy walk's analogous quantity is flight duration. The raw continuous-time
event stream (`levy_walk_events`) is not available to the isolated observer,
who only ever sees the gridded trajectory, so flights are reconstructed from
it (`_subgrid_flight_durations` in `inference/conditional.py`): flight speed
is estimated as `max(|diff|)` (no mixture of two flight directions can exceed
the true speed), each grid step is classified "pure" (interior to one flight)
or "straddling" (containing a turning point), and a straddling step's
fractional turning time is solved exactly from its one displacement value once
the incoming and outgoing directions are known from the neighbouring pure
steps. This sub-grid resolution is what makes the reconstruction usable at
all: `tau0` is order 1, the same order as `dt`, so a large fraction of true
flights are shorter than one grid step, and rounding turning points to the
nearest grid point (the naive detector) collapses exactly that short-duration
part of the distribution. A Hill estimator on the upper `top_frac` of the
reconstructed durations (`_hill_tail_exponent`) gives a tail-index estimate
`g_hat`; the final exponent is
`naive_weight * naive_alpha + (1 - naive_weight) * (3 - g_hat)`
(`estimate_exponent_levy_walk`), blended rather than using the Hill estimate
alone because its reliability depends on the flight count, which shrinks as
`g -> 1` (longer, fewer flights) and is not known in advance.

Ceiling check: a Pareto MLE on the *true*, unquantised event durations from
`levy_walk_events` reaches MAE ≈0.05 against naive's ≈0.13 on the same
held-out set — confirming the signal is real and substantial; the practical
grid-reconstructed version recovers only part of it.

**Tuning pitfall, recorded so it isn't repeated.** The first tuning pass fixed
`tau0 = speed = 1` (the `LevyWalkConfig` defaults) and found
`naive_weight = 0.6`, `top_frac = 0.4` cut MAE by ~20% relative, validated
across 8 independent seeds. Re-running Step 6 with real `draw_random_config`
draws (`tau0, speed ~ U(0.5, 2.0)`, as Step 6 actually samples) showed almost
no improvement with that first-pass tuning (0.1161 → 0.1163, i.e. very
slightly worse) — because a varying `tau0` puts more weight on the
harder-to-reconstruct short-flight regime than the fixed-`tau0 = 1` tuning set
did, and the tuned parameters had overfit to the easier case. Retuned against
600 realistically-drawn trajectories pooled over 10 seeds
(`naive_weight = 0.6`, `top_frac = 0.35`), validated on a fresh 300-trajectory
held-out set the tuning never saw (naive 0.133 → corrected 0.121, ~9%
relative), then re-ran Step 6 for real: `levy_walk` MAE 0.1161 → 0.1132
(overall `mae_oracle` 0.0478, `mae_predicted` 0.0527). **Lesson: tune and
validate against the actual parameter-generating distribution, not a
convenient fixed-parameter subset of it — even one other free parameter
(`tau0`) changed which correction looked best.**

**Flight-count diagnostic.** Across the interpolation range Step 6 actually
draws from (`gamma ~ U(1.2, 1.8)`, `tau0, speed ~ U(0.5, 2.0)`), the
reconstructed flight count never collapsed: min 32, median 218, max 305 over
the 150-trajectory `levy_walk` test set (`count_levy_walk_flights`, now
reported in `step6_conditional_exponent.json` as `levy_walk_flight_count`).
Checked separately across the full analytic range at fixed `tau0 = speed = 1`:
even at `gamma = 1.05` (near the extrapolation boundary, where flights run
longest and fewest), the minimum observed flight count over 15 trajectories
was 18 — comfortably above the Hill estimator's floor of 5.

**What would make this fragile.** If a future parameter sweep pushes `gamma`
close enough to 1 — or shortens `T` enough — that flight counts run low for a
non-negligible part of the range, `count_levy_walk_flights` should be checked
before trusting the correction there. The blend degrades toward the naive
estimate gracefully when the Hill fit is undefined outright (fewer than 5
usable durations), but a low-yet-nonzero flight count still feeds a noisy tail
estimate into the blend rather than being caught — it has not been stress-
tested outside the range Step 6 actually draws from.

## Phase 3 — Mean-Field Coupling

### Step 1, V1 — the J=0 reduction target was wrong (2026-09-06)

**Trigger.** Implementing `MeanFieldConfig` and `simulate_meanfield`
(truncated ARFIMA(0,d,0) velocity, `PROJECT.md` section 7) and validating V1
("reduction to Phase 1 at `J = 0`") against Davies-Harte fBm, as both
`PROJECT.md` and `PHASE3_PROMPT.md` specified: the increment ACF was off from
the fGn closed form by ~0.08-0.09 at lag 1 (e.g. 0.33 measured vs 0.41
predicted at `d = 0.25`), and — critically — this gap did **not** shrink
between `K = 64` and `K = 1024`, ruling out ordinary truncation error as the
cause before any tolerance could be picked.

**First (wrong) diagnosis: truncation.** The finite polynomial
`P_K(z) = 1 + sum_{s=1}^K c_s z^s` satisfies `P_K(z) -> (1-z)^d` only as
`K -> inf`, and `P_K(1)` (which should be exactly 0 in that limit) shrinks
only as `K^{-d}` — confirmed numerically exactly (`P_K(1)` at `K = 64, 256,
1024, 4096` scales by a factor `4^{-0.25} = 0.707` per 4x increase in `K`, to
the fourth decimal). This is a real, structural fact about the truncated
model — a finite `K` gives it a *finite* spectral density at zero frequency
instead of the true divergence, which in principle creates its own crossover
to ordinary diffusion at long enough lag, independent of `J` — and it is
documented in :func:`~collectivediff.dynamics.meanfield.truncated_arfima_acf`
because it is real. But it turned out not to be the explanation for the V1
gap: the lag-1 bias measured at `K = 64` and `K = 1024` was nearly identical,
whereas a `K^{-d}`-scaling artifact should have visibly shrunk over that 16x
range.

**Actual diagnosis, from the user: wrong reference process.** `PROJECT.md`
section 7 claimed the `J = 0` limit was "exactly fractional Gaussian noise."
It is exactly **ARFIMA(0,d,0)** instead. The two share the asymptotic
exponent `alpha = 1 + 2d` (both are `k^{2d-1}` at large lag) but have
different short-lag correlation: ARFIMA's exact closed form is
`rho(k) = Gamma(1-d)Gamma(k+d) / (Gamma(d)Gamma(k+1-d))`, giving
`rho(1) = d/(1-d)` — `0.3333` at `d = 0.25`, which is *exactly* what the
simulation measured (0.330-0.334 across five seeds) against fGn's `0.4142`.
Re-validating against this corrected target dropped the max ACF deviation
over the whole `[0, K/10]` window from ~0.08 to 0.005-0.007 — confirming
`simulate_meanfield` was correct all along, and the model derivation in
`PROJECT.md` section 7 had the wrong short-lag reference.

**A second, related correction: the in-window EA-MSD exponent is not
`1 + 2d`.** Convergence of ARFIMA's exact MSD to its asymptotic exponent is
fast but not instantaneous. An initial attempt to characterize how fast (via
a narrow local-slope estimator around specific lags) gave numbers that turned
out to be wrong by a wide margin and were not reproducible by three
independent methods (hand arithmetic on `Var(v_0 + v_1) = 2 + 2*rho(1)`, the
exact closed-form MSD summation, and fitting the actual simulated EA-MSD) —
all three agreed with each other and showed convergence to the asymptote
within a few percent by lag 10, not the ~9% gap first reported. Settled on the
cross-checked numbers:

| d | alpha(lag 10) | alpha(lag 100) | asymptote (1+2d) |
|---|---|---|---|
| 0.15 | 1.279 | 1.295 | 1.3 |
| 0.25 | 1.477 | 1.495 | 1.5 |
| 0.35 | 1.681 | 1.696 | 1.7 |
| 0.45 | 1.892 | 1.899 | 1.9 |

computed by `arfima_effective_exponent` (OLS fit of the exact closed-form MSD
over the window, not a bare asymptote assertion) — the correct target for
V1's EA-MSD check, and confirmed against the simulator: fitted exponents land
closer to this target than to `1 + 2d` itself (e.g. `d = -0.25`: fitted
0.571, exact in-window target 0.543, bare asymptote 0.5).

**What this means for the rest of the phase.** `d = 0.15` is close enough to
its asymptote's neighborhood that a short-window fit reads mostly like
Brownian motion (`alpha(10) = 1.279` against asymptote `1.3` looks fine in
isolation, but the *gap from 1.0* that the whole phase is trying to detect is
only `0.28`-`0.3`, thin against estimation noise) — worth keeping in mind
when Step 2 onward chooses which `d` values carry the sweep's statistical
weight. Also flagged, not yet resolved: the default `J` sweep in
`PHASE3_PROMPT.md` (`{0.05, 0.1, 0.2, 0.3, 0.5}`) puts most of its predicted
`tau_c = J^{-1/|d|}` values far outside the trustworthy `K/10` window at
`d = 0.25` (only `J = 0.5` lands `tau_c` inside it), which would starve V4's
`log(tau_c)` vs `log(J)` slope test of usable points — a recalibration
(choosing `J` from a target `tau_c` per `d`, e.g. `J = tau_c^{-|d|}` for
`tau_c in {8, 16, 32, 64}`) is planned for Step 1's remaining validations
(V2-V4) but not yet implemented or tested.

**Rejected fix, for the record.** Forcing `P_K(1) = 0` exactly (to remove the
truncation-driven finite-`S(0)` artifact noted above) was considered and
rejected: it gives the polynomial a *simple zero* at `z = 1`, so
`S(omega) ~ omega^{-2}` near zero frequency — a random walk in velocity, i.e.
`alpha -> 3` in position — not the fractional branch-point singularity
`(1-z)^d` a finite polynomial cannot reproduce at all. Confirmed numerically:
the corrected spectrum diverges and downstream MSD computations return NaN.
No polynomial fix of this kind is viable; the truncation artifact (real, but
small and living far outside `K/10`) is accepted and documented instead of
patched.

**Code:** `arfima_acf`, `arfima_effective_exponent`, `arfima_exact_msd`,
`truncated_arfima_acf`, `fractional_diff_kappa`, `simulate_meanfield` in
`dynamics/meanfield.py`; `MeanFieldConfig` in `config.py`. `PROJECT.md`
section 7 corrected in place (search "not fractional Gaussian noise" for the
amended paragraph). V1 tests in `tests/test_phase3_meanfield.py`; the
official Step 1 V1 numbers (`N=800, T=512, K=1024`) are in
`results/phase3/step1_v1_reduction.json`.

### Step 1, V2/V3 — sweep recalibration and deviation-spectrum normalisation (2026-09-07)

**Sweep recalibration.** `PHASE3_PROMPT.md`'s default `J` grid
(`{0.05, 0.1, 0.2, 0.3, 0.5}`) predicts `tau_c` values from 16 to 160000 at
`d = 0.25` — only `J = 0.5` lands inside the trustworthy `K / 10 = 102`
window at the default `K = 1024`. Replaced with a `tau_c`-targeted grid,
`J = tau_c^{-|d|}` for `tau_c in {8, 16, 32, 64}` plus `J = 0`
(`sweep_j_values`, `coupling_for_target_crossover` in
`dynamics/meanfield.py`), mirrored across the sign of `d` using `|d|` so the
negative-`d` invariance test (below) runs over comparable coupling
strengths, not comparable `tau_c` (which has no meaning for `d < 0`,
`PROJECT.md` section 7).

**Two cheap checks on the V1 numbers, both clean.** (1) The `d = 0` EA-MSD
exponent deviation from its exact target (1.0) shrank from 0.033 to 0.019
when `N` went `800 -> 3200` — roughly the `1/sqrt(N)` factor of 2 expected
from statistical noise (measured 1.68x), not a persistent fit-window bias.
(2) EB's *absolute* value at `d = 0` (0.150 at lag 51, `T = 512`) sits within
the same sampling-noise band as the already-validated Phase 1 Brownian
generator's own EB at matched `(N, T, lag)` (ratio to `brownian_eb` closed
form: 0.99-1.20 across four seeds for genuine Brownian motion, against 1.13
for the mean-field `d = 0` case) — no wrong constant factor in `eb_parameter`.

**V2 (conservation law): confirmed essentially exactly.** For every `d` in
`{±0.15, ±0.25, ±0.35, ±0.45}` swept across the recalibrated `J` grid with a
*shared* seed per `d`, the Whittle-fitted `d_hat` from the mean field agreed
across `J` to 1e-13 to 1e-15 — not approximately, to floating-point
precision. This is the expected behaviour, not a testing artefact: with the
same noise realisation reused across `J`, `mean_field(t+1) = mean(memory) +
J * (field_now - mean(v)) + mean(noise)`, and the middle term is `J` times
`(field_now - mean(v))`, which is 0 by the definition of `field_now` up to
floating-point rounding -- so the simulation reproduces the "`J` term cancels
exactly in the population average" claim bit-for-bit. `Var(mean_field)`
scaling check: measured ratio 9.31 against predicted 8.0 for `N = 200` vs
`1600` at one `(d, J)` point -- a single-realization variance-of-variance
estimate, not incompatible with `sigma^2 / N` given the effective sample size
of a persistent (`d = 0.25`) series is well under its raw length.

**V3 (deviation spectrum): a normalisation bug, then confirmed.** The first
periodogram-vs-`deviation_spectrum` comparison showed the two agreeing in
*shape* across a decade of frequencies but differing by a *flat* factor of
`2 pi` (ratio 0.150-0.165 against `1/(2 pi) = 0.159`) -- `deviation_spectrum`
had been written from the textbook AR spectral-density form
(`sigma^2 / |denominator|^2`), missing the extra `1 / (2 pi)` this codebase's
periodogram convention bakes in (`fgn_spectral_density`,
`whittle_log_likelihood`, both in `features/spectral.py`). Fixed by adding
the factor; re-checked at `K = 1024`, `N = 500`, `J = 0.3`, both signs of
`d`: mean absolute log-ratio 0.036-0.037 in the trustworthy band (typical
multiplicative deviation ~3.7%), for both `d = 0.25` and `d = -0.25`.

**Important negative finding, superseded below:** an early attempt at V4
(locating the memory crossover `tau_c` from the local slope of an agent's
EA-MSD) produced measured `tau_c` values that did not track the recalibrated
grid's targets at all (targets 8/16/32/64 -> measured 14.2/6.5/7.1/7.3, all
clustered regardless of target) and a `log(tau_c)` vs `log(J)` slope of
`+1.09` against a predicted `-4.0` -- wrong sign. Averaging 8 independent
realizations at a smaller scale ruled out noise as the cause: even smoothed,
the local exponent sat near 1.0-1.3 at the shortest lags, never approaching
the predicted `1 + 2d`, for every tested `J`. See the next entry for the
diagnosis and the replacement.

### Step 1, V4 — redesigned as a joint Whittle fit, not a crossover measurement (2026-09-07)

**Diagnosis.** The recalibrated `J` grid (chosen to keep `tau_c` inside
`K / 10`) makes `J` comparable to or larger than `kappa(1) = d` -- e.g. at
`d = 0.25`, `tau_c = 16` needs `J = 0.5`, twice `kappa(1) = 0.25`. The
`tau_c ~ J^{-1/|d|}` prediction comes from the *low-frequency* limit of
`S_delta` (`omega -> 0`, where `z^d -> 0` and `J` dominates the
denominator); it says nothing about the *short-lag* (broadband/high-frequency)
regime a local MSD slope actually probes, where `J`'s direct contribution to
the spectrum is not obviously subdominant to the memory term at all once `J`
is this large. The "runs `1 + 2d` below `tau_c`" half of the predicted
picture was never actually implied by the derivation for a `J` this strong --
it was an extrapolation beyond where the low-frequency argument applies, and
the measurement correctly refused to find something that was not really predicted.

**Redesign.** Replaced the crossover measurement with a joint `(d, J)`
Whittle MLE against `deviation_spectrum` (`whittle_dj_estimate` in
`dynamics/meanfield.py`): minimise
`sum[log S_delta(w; d, J) + I(w) / S_delta(w; d, J)]` over both parameters at
once, using the ensemble-averaged deviation periodogram (`_periodogram` in
`studies_phase3.py`, same one V3 already validated). This fits the whole
spectral shape rather than trying to read off one feature of it, and the
pass/fail criterion has no free parameter: does `J_hat` recover the `J` used
to generate the data, and does `d_hat` stay independent of `J`.

`J` is chosen from `J < |d| / 2` (memory dominant at short lags, opposite of
the old grid) rather than `tau_c`-targeted, since the crossover concept no
longer has a role in this test's design. `sweep_j_values` /
`RECALIBRATED_TAU_C` / `coupling_for_target_crossover` are kept -- they are
still correct and still used by V2, whose invariance test has no analogous
resolution problem.

**Optimizer bug found and fixed en route.** A single L-BFGS-B run started
from the midpoint of the search bounds converged to a poor local optimum
whenever the true `J` was far from that midpoint -- which was most of the
grid, since `J` is typically small. Concretely: `d = 0.35, J = 0` fit to
`d_hat = 0.380` from the bounds-midpoint start, but `d_hat = 0.354` (1%
error) from a start near the data's own scale. Fixed with a 3x3 multi-start
grid over `(d_0, J_0)`, keeping the lowest-objective result
(`whittle_dj_estimate`).

**Result, `K = 1024`, `N = 400`, `T = 1024`, `J < |d| / 2` grid, `d in
{0.15, 0.25, 0.35, 0.45}`:**

| d | d_hat mean | max \|d_hat - d\| | d_hat std across J |
|---|---|---|---|
| 0.15 | 0.1501 | 0.0022 | 0.0014 |
| 0.25 | 0.2493 | 0.0027 | 0.0023 |
| 0.35 | 0.3453 | 0.0084 | 0.0056 |
| 0.45 | 0.4467 | 0.0073 | 0.0058 |

`d_hat` is accurate to 1-2.5% and essentially `J`-independent for every `d`.
`J_hat` tracks the true `J` with *shrinking relative* error as `J` grows
(e.g. `d = 0.45`: true `J` 0.0225/0.0675/0.135 -> `J_hat` 0.0112/0.0593/0.126,
i.e. 50%/12%/7% relative error) and correctly reads `~0` at the `J = 0`
control -- weak coupling is harder to detect, which is expected statistical
behaviour, not a flaw.

**MSD crossover kept, demoted to illustration only.** `run_step1_v4_msd_illustration`
reproduces the original local-slope measurement at one point
(`d = 0.45, K = 16384, J in {0.05, 0.15, 0.3}`, `burn_in = 2K` rather than
`4K` to keep it affordable) purely for step 5's crossover figure. Its numbers
are not asserted against anything and must not be quoted as a measurement --
see the diagnosis above for why that measurement is unreliable at any
practically reachable `N`.

**Code:** `deviation_spectrum` (now periodogram-normalised),
`whittle_dj_estimate`, `coupling_for_target_crossover`, `sweep_j_values` in
`dynamics/meanfield.py` / `studies_phase3.py`. Tests in
`tests/test_phase3_meanfield.py` (`TestDeviationSpectrum`,
`TestCouplingForTargetCrossover`, `TestWhittleDjEstimate`, the latter
including a regression test pinned to the exact bad case the multi-start fix
was written for). Official numbers in `results/phase3/step1_v2_conservation_law.json`,
`step1_v3_deviation_spectrum.json`, `step1_v4_whittle_dj.json`,
`step1_v4_msd_illustration.json`.

### Step 2 — a naive Whittle "typical-agent" measure is misleading for d < 0 (2026-09-08)

**Trigger.** Directed to build step 2's two-regime table using Whittle
(V4's now-validated multi-start machinery) rather than an MSD slope, split
by sign of `d`, expecting: `d > 0` typical-agent regime falls toward 1 as
`J` grows while coherent stays at `1 + 2d`; `d < 0` both stay at `1 + 2d`
(coupling doesn't kill subdiffusive memory, `PROJECT.md` section 7).

**First attempt, wrong tool.** Fit the *joint* `(d, J)` model
(`whittle_dj_estimate`) to each agent's own `v_i`, ensemble-averaged.
Result: `d_hat` stayed flat at the *true* `d` across the entire `J` sweep,
for both signs -- which sounds like success but isn't what step 2 asks for.
Fitting the correct coupled model recovers the true, `J`-independent `d` by
construction (`d` and `J` are separate parameters in the model); that's
exactly what makes step 3's "model-aware" observer informative, but it
cannot show the phenomenological crossover step 2 exists to demonstrate,
since a well-specified model doesn't confuse suppressed memory with a
different `d` in the first place.

**Second attempt, better but still wrong for `d < 0`.** Switched to a naive,
coupling-blind Whittle fit (`whittle_d_only_estimate`, new: forces `J = 0` in
`deviation_spectrum`). For `d > 0` this correctly shows the expected decline
(e.g. `d = 0.25`: implied alpha 1.50 -> 0.74 across the sweep). But for
`d < 0` it *also* declined sharply (`d = -0.25`: alpha 0.50 -> 0.02;
`d = -0.45`: pinned at the fit's lower bound for `J >= 0.135`) --
contradicting the "subdiffusive memory survives" prediction. Root cause: the
`+ J e^{-i omega}` term in `deviation_spectrum` reshapes the *whole*
spectrum, not just its low-frequency limit (which is the only part the
`d > 0`/`d < 0` asymmetry argument is actually about) -- so forcing a
single-parameter model onto genuinely two-parameter data biases the fit
regardless of which sign of `d` the true process has. Confirmed spurious two
ways on the same simulated data: the model-aware fit stays flat at the true
`d` for `d < 0` too, and a **windowed** (not local -- V4's trap was about a
*local*, per-lag derivative, not a windowed fit) ensemble EA-MSD power-law
fit -- `fit_powerlaw` over `[2, K/10]`, `PROJECT.md`'s own TA/EA-MSD
convention, applied to the `N = 400`-agent ensemble average, hence not
noisy the way a single-trajectory local slope is -- also stays flat
(`d = -0.25`: exponent 0.55 -> 0.52) while still declining correctly for
`d = 0.25` (1.54 -> 1.18).

**Resolution.** `alpha_typical_agent` in the step 2 table is the windowed
EA-MSD fit -- a direct, model-free measurement of `Var(x_i(t))` growth,
which is what "exponent of the ensemble EA-MSD" (`PHASE3_PROMPT.md`'s own
phrase) literally means, and which cannot inherit a spectral model's
misspecification bias because it assumes no spectral model at all. Both
Whittle variants are kept in the table anyway
(`alpha_typical_agent_naive_whittle`, `*_model_aware`) because the
three-way contrast is itself the finding: naive Whittle shows a
plausible-looking but wrong story (decline on both signs); model-aware
recovers the true, sign-symmetric `d` on both signs (unsurprising once you
see it's fitting the right model); windowed MSD recovers the true,
*asymmetric* phenomenology the section 7 prediction is actually about.
`whittle_d_only_estimate`'s docstring now states this limitation directly
rather than the original (wrong) "biased toward 1" characterization --
the bias runs toward antipersistence regardless of the true sign, not
toward any particular alpha.

**Confirmed sign asymmetry, all 8 `d` values, windowed EA-MSD (**
`N=400, T=1024, K=1024`**, full table in
`results/phase3/step2_two_regimes.json`):** for every `d > 0` in
`{0.15, 0.25, 0.35, 0.45}`, `alpha_typical_agent` falls monotonically across
the `J` sweep (e.g. `d=0.45`: 1.86 -> 1.34); for every `d < 0` in
`{-0.15, -0.25, -0.35, -0.45}` it stays within about 0.05-0.1 of its `J=0`
value across the same sweep. `alpha_coherent` is flat across `J` for every
`d` (V2, reconfirmed), though offset from the bare `1 + 2d` asymptote by an
amount that grows as `|d|` approaches the `(-0.5, 0.5)` boundary (e.g.
`d=-0.45`: coherent 0.367 vs asymptote 0.10) -- consistent with the known
degradation of Whittle/MLE behaviour for long-memory parameters near the
edge of the parameter space, not a new bug; not investigated further here.

**Code:** `whittle_d_only_estimate` (`dynamics/meanfield.py`),
`_one_dj_step2_point` / `run_step2_two_regimes` (`studies_phase3.py`). Tests:
`TestWhittleDOnlyEstimate` (parametrized both signs),
`TestWindowedEaMsdRegimeAsymmetry` in `tests/test_phase3_meanfield.py`.

### Steps 3-5 (2026-09-08)

**Step 3 — three observers.** Isolated model-agnostic (Phase 2's classifier +
correction pipeline, reused unmodified, on `x_i`), isolated model-aware
(`whittle_dj_estimate` on one agent's own `v_i`), field-aware
(`whittle_d_estimate` on `<v>`), and an agent+field combination (simple
average of model-aware and field). Bias/variance pooled over 20 sampled
agents x 10 realizations per `(d, J)` point (`N=400, T=1024, K=1024`, full
numbers in `results/phase3/step3_observers.json`):

- **Model-agnostic**: bias grows sharply with `J`, in the *same direction*
  regardless of the sign of `d` (`d=0.25`: -0.05 -> -0.37; `d=-0.25`:
  +0.07 -> -0.14) -- consistent with step 2's finding that naive,
  coupling-blind estimators get pulled toward antipersistence by `J`
  regardless of the true regime. Variance stays small and roughly flat
  (~0.0004-0.0006) throughout: precise but increasingly wrong, the same
  "precise and inaccurate" failure mode Phase 1 documented for scaled
  Brownian motion.
- **Model-aware**: bias stays small (|bias| < 0.06 throughout both signs)
  but variance grows with `J` (0.0075 -> 0.0155 at `d=0.25`) and is the
  largest of the four observers at every point -- correctly unbiased,
  but a single agent's own periodogram is a genuinely noisy basis for a
  two-parameter fit, more so as `J` grows.
- **Field-aware**: bias and variance both exactly flat across `J` (down to
  the last reported digit) -- the conservation law again, now visible in an
  estimator's behaviour rather than in the mean field's own spectrum.
- **Combination**: bias and variance both sit between model-aware and
  field-aware, closer to field -- averaging a noisy-but-unbiased estimate
  with a precise one pulls the combination toward the precise one, as
  expected.

**V5, done properly.** The main sweep's "isolated model-aware" observer
fits two parameters `(d, J)` per agent (needed for `J > 0`), while
field-aware fits one (`<v>` has no `J`); comparing their variances directly
conflates "does averaging over `N` help" with "is a 2-parameter fit noisier
than a 1-parameter fit," and the first naive comparison showed isolated
~9x noisier than field -- almost entirely the latter effect, not the former.
`run_step3_v5_check` reruns *only* at `J=0`, where both sides can
legitimately use the same one-parameter model
(`whittle_d_only_estimate`, correctly specified there). Result
(`N=400, T=1024, K=1024`, 15 realizations x 20 agents,
`results/phase3/step3_v5_check.json`): variance ratio (isolated / field)
3.6x at `d=0.25`, 3.9x at `d=-0.25` -- comparable in order of magnitude, not
the ~400x an `N`-fold precision gain would predict. Confirms the analytic
argument in `run_step3_observers`'s docstring: Whittle/MLE precision for a
spectral *shape* parameter doesn't depend on the spectral density's overall
scale, so averaging away `1/N` of the *amplitude* doesn't buy `1/N` of the
*variance* on `d`. The residual ~3.6-3.9x (not ~1x) is plausibly finite-`T`
correction plus real sampling noise in the field-variance estimate itself
(only 15 realizations); not chased further.

**Step 4 — Q3 correlation.** `d ~ U(-0.4, 0.4)`, 30 realizations x 5 sampled
agents per `J in {0, 0.1, 0.3, 0.5}` (`results/phase3/step4_q3_correlation.json`).
All four observers correlate strongly with true coherent `d` (0.93-0.99) --
unsurprising given `d` is drawn from a wide range, which any estimator that
gets the *sign* right will correlate well with. The differentiation that
does show up matches step 3: field-aware highest and flattest (~0.991
throughout); model-agnostic close behind and essentially flat (~0.98-0.985,
the systematic bias found in step 3 apparently stays too smooth/monotonic in
`d` to hurt correlation much); model-aware declines most with `J` (0.973 ->
0.932, tracking its growing variance); combination between field and
model-aware, closer to field.

**Step 5 — figures.** `viz/phase3.py`: `plot_crossover_illustration` (fig 1,
reads `step1_v4_msd_illustration.json`, now extended to also save the
coherent mode's own local exponent for the same realization -- expect it
noisy, there is only one center-of-mass trajectory per run, unlike the
ensemble-averaged agent curve; kept qualitative per the item's own
documented limits), `plot_observer_bias_variance` (fig 2, step 3),
`plot_q3_correlation` (fig 3, step 4, y-axis zoomed to the actual 0.93-1.0
range the data lives in -- the full [0,1] axis made every curve look flat).
`animate_meanfield_comparison` (fig 4): two independent 1-D mean-field runs
combined as x/y components (`PROJECT.md` section 7's explicit 2-D
convention), `N=60, T=400, K=512` for a legible, cheap animation -- not a
measurement, so no need for the full-scale `K`. All four in
`results/media/phase3_figures/`. Tests in `tests/test_phase3_studies.py`
(structural/plumbing checks at tiny scale; the physics is validated at full
scale above and in `test_phase3_meanfield.py`).

**Timing** (`N=400, K=1024, T=1024` unless noted; wall-clock, single machine,
`rtk proxy`): one simulation ~3.3s; one isolated-observer agent analysis
(model-agnostic or model-aware) ~0.05-0.06s; classifier training (one-off,
per trajectory length) ~8s. Full-study wall times: step 2 (64 `(d,J)`
points) 226s; step 3 (160 sims + 3200 agent analyses) 1342s; step 3 V5 check
(30 sims + 600 agent analyses) 98s; step 4 (120 sims + 600 agent analyses)
534s; step 1 V4 MSD illustration at `K=16384, N=150` (3 `J` values,
`burn_in=2K`) 1078s (~360s/simulation at that `K`) -- confirms
`PHASE3_PROMPT.md`'s own expectation that `O(NK)` cost is the real budget
constraint for phase 4 to inherit: cost scales with `K` far faster than with
`N` (16x `K` cost ~110x, well above linear, from the burn-in length scaling
with `K` on top of the per-step `O(NK)` term).

**Scope not run:** the full `N in {100,1000} x T in {1024,8192}` x every
`d`/`J` combination sweep `PHASE3_PROMPT.md` describes as the eventual
target. Every study above used one representative, moderate scale
(`N=400, T=1024, K=1024`, occasionally smaller for speed) rather than the
full Cartesian product, which at the measured per-simulation cost would run
into hours. Steps 2-4's numbers should be read as validated at this scale,
not as the phase's final large-`N`/large-`T` results.

### Pre-phase-4 close-out: two more bugs, both in the field-side Whittle fit (2026-09-08)

**Item 1 -- V5's ratio should have been 1.00, not 3.6-3.9.** Directed to check
this against the analytic prediction: Whittle Fisher information for
ARFIMA(0,d,0)'s `d` is `pi^2 / 6` per observation, independent of `d` and of
scale (verified numerically against the closed-form integral,
`1.64497` vs `pi^2/6 = 1.64493`), so an efficient single-agent fit and an
efficient field fit at `T=1024` should both sit near the CRLB
`6/(T pi^2) = 5.94e-4`, giving a variance ratio of 1.00. The measured 3.6-3.9
meant one arm was roughly 4x less efficient -- investigated and found to be
**two separate bugs**, not one:

1. **Missing scale profiling in the new ARFIMA Whittle fits.**
   `whittle_dj_estimate` / `whittle_d_only_estimate` (`dynamics/meanfield.py`)
   called `deviation_spectrum` at its default `sigma = 1` without ever
   fitting the noise scale -- fine for an individual agent (true scale
   exactly 1, matching the config) but catastrophic for the mean field
   (true scale `sigma^2 / N`): applying the unprofiled fit directly to
   `<v>` pinned `d_hat` at the search bound (`0.49`) with near-zero
   variance, not because it was precise but because it was degenerate.
   Fixed by profiling `sigma^2` out analytically
   (`_profiled_whittle_objective`), exactly as
   `whittle_mle_hurst` already does for fGn -- valid because `sigma` is a
   pure multiplicative prefactor of `deviation_spectrum` for *any* fixed
   `(d, J)`, so eliminating it doesn't change what `(d, J)` minimises the
   objective.

2. **`whittle_d_estimate` fit the wrong model -- the actual reason for the
   original bad ratio.** This function (`studies_phase3.py`, used for
   *every* "coherent"/"field" number in the phase so far: V2, step 2's
   coherent regime, step 3's field-aware observer, step 4's field
   correlations) called `whittle_mle_hurst`, which fits **fGn**, not
   ARFIMA. `<v>` is exactly ARFIMA(0,d,0) (`PROJECT.md` section 7) --
   fitting fGn to it is the identical mistake V1 already found and fixed
   for a single agent (step 1), just never caught on the field side,
   because none of V2/step 2/step 3/step 4 checked the field's fitted
   variance against its analytic Whittle efficiency until V5 did. The tell
   was unambiguous: fGn-fit field variance came out to about a quarter of
   the ARFIMA CRLB, `efficiency > 1`, a mathematical impossibility for a
   correctly-specified unbiased estimator. Fixed by rewriting
   `whittle_d_estimate` to call `whittle_d_only_estimate` (ARFIMA, `J=0`,
   now correctly profiled) instead.

**Result after both fixes**, `N=400, T=1024, K=1024`, 40 realizations x 20
agents (`results/phase3/step3_v5_check.json`):

| d | isolated efficiency | field efficiency | ratio (isolated/field) |
|---|---|---|---|
| 0.25 | 0.987 | 1.092 | 1.106 |
| -0.25 | 0.987 | 1.074 | 1.088 |

Both arms sit within ~10% of the CRLB, and the ratio sits within ~10% of
1.00 -- matching the analytic prediction, not the 400x an `N`-fold precision
gain would give (the substantive part of V5 stands, now on the numbers it
should have had). `field_efficiency` slightly above 1 (not exactly 1) is
plausibly finite-`T`/finite-sample (`n=40`) noise; not chased further.

**Downstream consequence, found immediately on re-running step 2:** the
large, previously-unexplained offset between `alpha_coherent` and the bare
`1+2d` asymptote (e.g. `d=-0.45`: 0.367 vs 0.10, a +0.267 gap) was
attributed at the time to "known degradation of Whittle/MLE behaviour near
the `d` boundary." That explanation was wrong -- it was bug 2. Re-run with
the corrected field estimator: `d=-0.45` coherent is now 0.103 (gap +0.003);
`d=0.45` coherent is now 1.866 (gap -0.034, was -0.192). The boundary-effect
explanation is retracted. Every "coherent"/"field" number reported before
this entry (V2's `d_hat_field`, step 2's `alpha_coherent`, step 3's
`field_aware`/`combination_agent_field`, step 4's field correlations) used
the misspecified fGn fit and needed re-running; step 2 is done (numbers
above), steps 3 and 4 are being re-run and superseded below.

**Item 3 -- single-agent model-aware fits genuinely pin at the `d` boundary
some of the time, and this is not an optimizer bug.** Investigated whether
step 3's finding (model-aware the *worst* isolated observer by RMSE at weak
`J`) was a residual local-optimum problem in `whittle_dj_estimate`'s
multi-start, the natural place to check given V4's earlier optimizer bug.
It is not: sampled 60 individual agents at one `(d, J)` point and found 8
(13%) with `d_hat` pinned at the upper bound `0.49`; for one such case,
scanning the profiled objective along `d` at the fitted `J` shows it
strictly decreasing (better) all the way to the bound -- a genuine
boundary-seeking optimum for that noise realization, not a missed interior
one. Excluding the pinned cases drops the sample's `d_hat` std from `0.104`
to `0.072` (a ~30% reduction) and pulls the mean from `0.288` back to
`0.257` (true `0.25`) -- consistent with a real, if uncomfortably frequent,
finite-sample identifiability limit of fitting two parameters `(d, J)` from
one noisy periodogram, not a bug to fix. Step 3 is being re-run with the
corrected field estimator (item 1/2's fix); the model-agnostic-vs-model-aware
RMSE ordering will be re-examined against the corrected numbers below rather
than the pre-fix ones once that run completes.

### Pre-phase-4 close-out, completed (2026-09-10)

The four items above are now closed, in the sense of "measured and reported"
-- not all four came back clean. Item 1's numbers were already right in the
previous entry; items 2 and 4b are genuinely clean; item 3 turned out to be
worse than the Sept-8 entry claimed for `d < 0`, and item 4a surfaces a real,
unexplained anomaly. Reporting all four together as directed.

**Item 1 -- V5 ratio.** Already fixed by the two bugs in the entry above;
nothing new to run. Restated for completeness,
`results/phase3/step3_v5_check.json` (`N=400, T=1024, K=1024`, 40
realizations x 20 agents): `d=0.25` -> isolated efficiency 0.987, field
efficiency 1.092, ratio 1.106; `d=-0.25` -> 0.987 / 1.074 / 1.088. Both arms
sit within ~10% of the CRLB `6/(T pi^2) = 5.94e-4`; the ratio sits within
~10% of the analytic prediction of 1.00. This is the supervisor's named gate
for Phase 4, and it is cleared.

**Item 2 -- Q3 metric.** Already fixed (NRMSE, `1 - R^2` reported alongside
correlation, `run_step4_q3_correlation`'s `metrics()`). The numbers now
actually separate the four observers, which raw correlation never did
(`results/phase3/step4_q3_correlation.json`, `d ~ U(-0.4, 0.4)`, 30 draws x 5
agents per `J`, nrmse/r2 per observer):

| J | agnostic | model-aware | field | combo |
|---|---|---|---|---|
| 0.0 | 0.301 / 0.909 | 0.268 / 0.928 | 0.135 / 0.981 | 0.160 / 0.974 |
| 0.1 | 0.376 / 0.858 | 0.330 / 0.890 | 0.135 / 0.981 | 0.185 / 0.966 |
| 0.3 | 0.757 / 0.424 | 0.536 / 0.711 | 0.135 / 0.981 | 0.277 / 0.923 |
| 0.5 | 1.131 / -0.289 | 0.529 / 0.718 | 0.135 / 0.981 | 0.285 / 0.918 |

Field-aware is exactly flat across `J` (the conservation law again -- `<v>`
doesn't know `J` exists). Model-agnostic degrades the worst: by `J=0.5` its
`R^2` is negative, i.e. worse than just predicting the mean `d` every time.
Model-aware degrades too but less severely, tracking step 3's
growing-variance finding. Correlation alone (0.93-0.99 everywhere, still
reported for continuity) would have hidden all of this -- confirming the
diagnosis that motivated switching metrics in the first place.

**Item 3 -- observer ordering, formalised and partly retracted.**
`run_step3_residual_surface` (new; `dynamics/meanfield.py`'s
`_profiled_whittle_objective` exported as `whittle_objective` for this) turns
the Sept-8 ad hoc 60-agent check into a saved artifact
(`results/phase3/step3_residual_surface.json`). That check examined one
`(d, J)` point without recording its sign and concluded, in general, "a
genuine ... finite-sample identifiability limit ... not a bug to fix." Its
own numbers (8/60 pinned, mean pulled from 0.288 to 0.257) line up closely
with this entry's `d=0.25` point below, so it was very likely describing the
`d > 0` regime specifically and was right about it -- but generalised past
what it had actually checked. **The `d < 0` side, checked here for the first
time, is a different story.**

Two points, both members of `combined_j_sweep`'s strong grid, chosen as the
largest isolated-model-aware variance (`d=0.25`) and largest bias
(`d=-0.25`) in `step3_observers.json`:

- `d=0.25, J=0.5946` (`tau_c=8`): 10/60 agents (16.7%) pin at the `d_hat`
  bound `0.49`. A 41x41 grid over the whole `(d,J)` box, evaluated at the
  most boundary-adjacent agent, has its global minimum within 2 grid cells
  of the multi-start optimizer's own answer
  (`grid_matches_optimizer_result = True`), and scanning the objective along
  `d` at the fitted `J` is monotonic all the way to the bound. Refitting all
  60 agents with a much denser 7x7=49-start grid changes **zero** of the 60
  fits by more than 0.02. This side is exactly what Sept-8 described: a
  genuine, unfixable-by-more-optimisation finite-sample wall.
- `d=-0.25, J=0.3536` (`tau_c=64`): 11/60 (18.3%) pin at `d_hat = -0.49`.
  Here the 41x41 grid's global minimum (`d~-0.29, J~0.30`) is *not* near the
  optimizer's boundary answer (`grid_matches_optimizer_result = False`) and
  scores 1.32 nats better. A targeted 81-start refinement from that grid
  point converges to `d~-0.311, J~0.275` at objective -935.56 against the
  shipped 9-start fit's -933.99 (likelihood ratio ~4.8x) -- close to the true
  `(d,J)=(-0.25, 0.354)`, not the boundary. Refitting all 60 agents with the
  same denser 49-start grid: 29/60 (48%) of fits change by more than 0.02,
  mean bias roughly halves (-0.148 -> -0.070), variance is essentially
  unchanged (0.0082 -> 0.0093, within noise), and the pinned-at-bound count
  drops from 11 to 6 (some genuine pins remain even under the denser
  search).

So the two candidate causes the supervisor asked to separate are **both
real, but on opposite signs of `d`**: `d > 0`'s poor isolated-model-aware
performance is the honest variance cost of a two-parameter fit on one noisy
periodogram; `d < 0`'s is *substantially* the 3x3=9-start multi-start in
`whittle_dj_estimate` landing in the wrong basin, not (only) an
identifiability wall. This is a real, fixable bug the Sept-8 entry missed
because it checked only a `d`-only scan at the fitted `J`, not a full grid --
exactly what a genuine 2-D residual surface catches that a 1-D scan doesn't.

**Not fixed here, deliberately.** Widening the multi-start grid in
`whittle_dj_estimate` would touch every downstream number that calls it --
step 1 V4, step 2's `*_model_aware` columns, step 3's isolated model-aware
observer, and the `d<0` half of step 4's Q3 table above -- and re-validating
that whole cascade is a bigger undertaking than this close-out's scope.
Flagged, not silently skipped, matching this file's own precedent
(PHASE1_REVIEW.md item 5's deferred refactor): **`whittle_dj_estimate`'s
multi-start should be widened (or re-targeted toward large-`J`/negative-`d`
corners) before Phase 4 leans on the isolated model-aware observer for
`d < 0`, strong-`J` populations**, and every existing `*_model_aware` number
for `d<0` in steps 1-4 should be read as an upper bound on that observer's
true bias, not its final value.

**Corroborating, pre-existing test failure.** Running the full slow suite
for this close-out surfaced
`TestWhittleDOnlyEstimate::test_biased_by_coupling_regardless_of_true_sign[-0.25]`
(`tests/test_phase3_meanfield.py`) already red, deterministically, at
`d=-0.25, J=0.35, seed=41` -- `d_aware` comes out at `-0.408` instead of near
the true `-0.25`, failing the test's `d_naive < d_aware - 0.1` assertion
because the two nearly coincide (`-0.409` vs `-0.408`). This is not a
regression from this close-out (no code touched by this entry changes
`whittle_dj_estimate`'s or `whittle_d_only_estimate`'s behaviour -- only a
new pass-through wrapper, `whittle_objective`, was added) and it is not
fixed here, for the same cascade-scope reason as above. It is independent,
unit-test-level confirmation of exactly this section's finding: this
specific `(d, J, seed)` is another case of the model-aware multi-start
landing in a bad basin for `d < 0`. Left red deliberately rather than
patched or marked `xfail` -- weakening the assertion would enshrine the bug
as expected behaviour instead of flagging it.

**Item 4a -- N-scaling of field variance: real, and unexplained.**
`run_step4_n_scaling` at `n_realizations=10` showed `field_variance` rising
monotonically with `N` (0.000248 -> 0.000340 -> 0.000417 -> 0.000555 across
`N = 50, 100, 400, 1600`, identically at `J=0.0` and `J=0.3` -- expected,
the conservation law makes the field exactly `J`-independent). This
shouldn't happen: `n_steps` is fixed across the sweep and Whittle precision
for a spectral *shape* parameter is scale-invariant (V5's whole point), so
`Var(d_hat_field)` should not depend on `N` at all.

`run_step4_field_variance_vs_n` (new) reruns at `J=0` only (justified above)
with `n_realizations=20` instead of splitting the same budget across two
`J` values, for a tighter estimate
(`results/phase3/step4_field_variance_vs_n.json`):

| N | variance | 95% CI | fraction near d-bound |
|---|---|---|---|
| 50 | 0.000383 | [0.000222, 0.000817] | 0.000 |
| 100 | 0.000544 | [0.000315, 0.001160] | 0.000 |
| 400 | 0.000812 | [0.000469, 0.001731] | 0.000 |
| 1600 | 0.000986 | [0.000570, 0.002104] | 0.000 |

The trend not only survives doubling the realization count, it's slightly
*stronger* (ratio `N=1600`/`N=50` now 2.57x, up from 2.24x at 10
realizations) and remains perfectly monotonic across all four points in
both runs independently. An F-test on the two extreme variances
(`var[1600]/var[50] = 2.57`, 19 and 19 d.o.f.) gives `p = 0.046` -- a real,
if modest, effect at conventional significance, not noise. Log-log slope
across the four points is `~0.27` (`var ~ N^0.27`): clearly rising, clearly
not the flat line V5 and the conservation law predict, but far short of the
`N^{-1}` the *typical-agent* MSD estimator shows in the same table (see
below) -- this is not a fold of the "more agents help" effect leaking into
the field estimator, it is something smaller and separate.

The candidate mechanism checked and **ruled out**: more `d_hat_field` draws
landing near the search bound at large `N` (would show up as boundary
pinning inflating the sample variance, the same failure mode item 3 found
for the isolated model-aware observer). `fraction_near_d_bound` is exactly
`0.000` at every `N` -- no pinning anywhere in this sweep. Whatever is
driving the rise, it is not that.

**Not resolved.** Following the instruction not to force an explanation
that doesn't fit the data: this is flagged, not explained. It survived a
realization-count check and a specific candidate-mechanism check; it has not
survived a root-cause investigation, which would need to look inside
`whittle_d_only_estimate`'s optimizer behaviour or the periodogram
construction at large `N` (e.g. whether `_periodogram`'s frequency grid or
floating-point cancellation in `mean_field`'s construction degrades subtly
as individual-agent amplitude/`N` grows) -- out of scope for this close-out.
**Flagged for Phase 4**: the field-aware observer's precision should not be
assumed exactly `N`-independent when budgeting; treat the CRLB efficiency
figures above (V5, `N=400`) as calibrated at that one `N`, not guaranteed to
hold at every scale Phase 4 might use.

**Item 4b -- K-scaling.** `run_step4_k_scaling` (new), `N=50, T=64,
burn_in=2K`, one run per `K` (`results/phase3/step4_k_scaling.json`). This
table is the clean re-run, with nothing else on the machine competing for
CPU; a first attempt run concurrently with another background job gave
`K=16384` elapsed times of 279-378s (noted below) purely from contention,
not from the algorithm -- a reminder that these single-run wall-clock
numbers are only meaningful uncontended, matching every other timing note
in this file:

| K | burn_in | elapsed | ratio over prior K (4x increase) |
|---|---|---|---|
| 256 | 512 | 0.025s | -- |
| 1024 | 2048 | 0.216s | 8.7x |
| 4096 | 8192 | 2.179s | 10.1x |
| 16384 | 32768 | 117.3s | **53.8x** |

Fitted log-log slope `p = 2.00`. The naive algorithmic baseline is not
`p=1` here: burn-in itself scales as `O(K)`, so `total_steps ~ K` and
per-step cost `~ O(NK)` already predicts `p=2` before any cache effect --
and the overall fitted slope lands almost exactly on that baseline. The
`256->1024->4096` ratios (8.7x, 10.1x) sit *below* the 16x quadratic
prediction (small-`K` fixed Python-loop overhead dominates over the `O(NK)`
matmul there, so it doesn't scale down as fast as the asymptotic form
suggests), which is what pulls the overall `p` down to 2.00 despite the
sharp last step: `4096 -> 16384` alone is 53.8x against a 16x quadratic
prediction -- a real, localised ~3.4x excess, not explained by complexity
alone, consistent with the `(N, K)` ring buffer (the per-agent memory
window in `simulate_meanfield`) outgrowing cache specifically at that
transition rather than a smooth trend across the whole sweep. **Caps
Phase 4's feasible `K`**: `K=16384` costs ~117s per simulation even at this
tiny `N=50`, and that single point accounts for 98% of the whole 4-point
sweep's total wall time; anything beyond `K~4096` should be budgeted with
this localised jump in mind, not a smooth `O(NK^2)` extrapolation.

A separate solo timing check (not part of `run_step4_k_scaling`, which
holds `N` fixed) found the same qualitative effect on the *other* axis: at
fixed `K=1024, T=1024`, `N=1600` costs 79.1s wall-clock against `N=400`'s
5.0s -- 15.8x for a 4x `N` increase, well above the `O(N)` baseline (4x)
that fixed-`K` scaling predicts. This is a *wall-clock cost* finding, not to
be confused with item 4a's *estimator-variance* finding above -- the two are
about different quantities and there is no evidence linking them (item 4a's
`fraction_near_d_bound` check ruled out the one mechanism that could have
connected a timing/numerical artifact to a variance artifact). Read as its
own result: `simulate_meanfield`'s per-step cost and memory footprint are
driven by the `(N, K)` ring buffer's *product*, not either dimension
independently, so Phase 4's agent-count budget and Phase 3's memory-length
budget draw on the same cache resource for wall-clock cost, even though
(per item 4a) they do not obviously share a mechanism for estimator
variance.

**Verdict.** Item 1, the supervisor's explicit gate ("Phase 4 does not start
until item 1 is resolved"), is resolved -- Phase 4 is not blocked on it.
Items 2 and 4b are cleanly closed: the Q3 metric now actually discriminates
observers, and the K-scaling cost is measured and characterised (roughly
quadratic overall, but with a sharp, localised cache jump at `K=16384` that
caps `K` well below there in practice). Items 3 and 4a are closed *with
findings* that should inform Phase 4's design rather than be read as green:

- **Item 3**: a real, unfixed bug -- `whittle_dj_estimate`'s 9-start
  multi-start misses a materially better optimum for roughly half of `d<0`,
  strong-`J` individual-agent fits. Not fixed here (would invalidate a wide
  downstream cascade); every existing `d<0` `*_model_aware` number in this
  file should be read as an upper bound on that observer's true bias.
- **Item 4a**: a real, statistically detectable (`p=0.046`), unexplained
  rise in field-aware variance with `N` (`~N^0.27`), contradicting the flat
  prediction V5 established at `N=400`. One candidate mechanism was ruled
  out; the true cause is still open.

Recommend before Phase 4 leans on either mechanism: (1) widen
`whittle_dj_estimate`'s multi-start grid (or re-target it toward
large-`J`/negative-`d` starts) and re-run the affected `model_aware` numbers
in steps 1-4; (2) either re-derive why field variance rises with `N` or
treat the V5 efficiency numbers as valid only near `N=400` until it is
understood; (3) budget any `(N, K)` sweep with the localised cache jump at
large `K` (and the milder but real super-linear `N`-cost) in mind -- a naive
smooth `O(NK^2)`/`O(N)` extrapolation will underestimate wall-clock cost
badly once either axis is pushed past roughly `K~4096` or `N~1600` at this
machine's cache sizes.

### Item 3 bug fixed: widened `whittle_dj_estimate` multi-start (2026-09-12)

The Sept-10 entry's "Not fixed here, deliberately" paragraph is superseded by
this entry -- left in place above as an accurate record of what was true at
the time, not deleted. Every `d<0` `*_model_aware` number quoted before this
entry should now be read against the corrected numbers below, not treated as
an upper bound going forward.

**Fix.** `whittle_dj_estimate`'s multi-start grid (`dynamics/meanfield.py`)
widened from 3x3=9 starts (`d0 in (-0.2, 0.0, 0.2)`, `j0 in (1e-3, 0.05,
0.3)`) to 7x7=49 starts: `d0 in (-0.4, -0.267, -0.133, 0.0, 0.133, 0.267,
0.4)`, `j0 in (1e-3, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0)`. The old grid's most
negative `d0` (`-0.2`) and largest `j0` (`0.3`) sat right at the edge of the
basin the residual-surface diagnostic had already located
(`d~-0.31, J~0.275-0.3`) with nothing beyond either edge to descend from.
Docstring updated in place with a "Second multi-start fix" paragraph.

**Direct confirmation on the known-bad case.** `d=-0.25, J=0.35, seed=41`,
ensemble periodogram (`N=400, T=1024, K=1024`, the exact data
`test_biased_by_coupling_regardless_of_true_sign[-0.25]` uses): `d_hat`
before `-0.408` (indistinguishable from the naive coupling-blind fit's
`-0.409`), after `-0.248` (true `-0.25`) with `J_hat=0.349` (true `0.35`).
`pytest tests/test_phase3_meanfield.py -k "TestWhittleDjEstimate or
TestWhittleDOnlyEstimate" -m slow` -- 5/5 pass, including the previously-red
parametrization. Full slow suite for `tests/test_phase3_meanfield.py` and
`tests/test_phase3_studies.py`: 19 passed (0 failed); fast suite for the same
two files (`-m "not slow"`): 16 passed. Nothing else regressed.

**Residual-surface diagnostic, rerun with the new grid**
(`results/phase3/step3_residual_surface.json`, same two points, same 60
agents, same seeds):

| point | pinned before | pinned after | bias (incl. pinned) before -> after | `grid_matches_optimizer_result` |
|---|---|---|---|---|
| `d=0.25, J=0.5946` | 10/60 | 10/60 | 0.0201 -> 0.0201 | `True` (unchanged) |
| `d=-0.25, J=0.3536` | 11/60 | 6/60 | -0.1480 -> -0.0703 | `False` -> `True` |

The `d>0` point is exactly as before: same pin count, same bias, grid still
agrees with the optimizer. The `d<0` point's pinned count drops as predicted
and `grid_matches_optimizer_result` flips to `True` -- the 49-start grid now
finds the same optimum the exhaustive 41x41 box search finds, closing the
gap the Sept-10 diagnostic itself measured (`-935.56` vs `-933.99`).

**Cascade, re-run at full scale** (`run_step1_v4_whittle_dj`,
`run_step2_two_regimes`, `run_step3_observers`, `run_step4_q3_correlation`,
`run_step3_residual_surface`, `run_step4_n_scaling`, all default params
confirmed on disk: `N=400, T=1024, K=1024` throughout, `n_realizations`/
`n_draws`/`n_values` all matching each function's documented defaults):

- **Step 3 isolated model-aware, `d=-0.25`** (`step3_observers.json`, bias /
  variance, before -> after): `J=0.3536`: `-0.1259/0.00953` ->
  `-0.0518/0.00798`; `J=0.4204`: `-0.1218/0.00523` -> `-0.0489/0.00637`;
  `J=0.5000`: `-0.0973/0.00382` -> `-0.0415/0.00478`; `J=0.5946`:
  `-0.0334/0.00356` -> `-0.0294/0.00359`. Bias roughly halved at the three
  strongest couplings; variance moves within noise (up at two points, down
  at one) -- consistent with the fit now landing in the correct basin more
  often rather than becoming intrinsically less noisy. Weak-`J` points
  (`<=0.075`, where the old grid was already fine) are unchanged to within
  Monte Carlo noise, as expected. **`d=0.25` side is unchanged in substance**
  (e.g. `J=0`: bias `0.0558->0.0580`, variance `0.0076->0.00792`) -- small
  shifts consistent with a handful of individual fits moving by less than
  the diagnostic's 0.02 threshold, not a basin change.
- **Step 4 Q3, isolated model-aware** (`step4_q3_correlation.json`, `d ~
  U(-0.4, 0.4)` so only part of the sample is on the buggy side): nrmse/r2
  before -> after: `J=0.0`: `0.268/0.928 -> 0.295/0.913` (essentially flat,
  within the redraw-free noise floor of a fit that barely changes for most
  agents at weak coupling); `J=0.1`: `0.330/0.890 -> 0.330/0.890`; `J=0.3`:
  `0.536/0.711 -> 0.488/0.760` (improved); `J=0.5`: `0.529/0.718 ->
  0.519/0.729` (mildly improved). `field_aware` is bit-for-bit unchanged
  (`0.1346/0.9813` at every `J`, as it must be -- it never calls
  `whittle_dj_estimate`), confirming the movement is coming from the fixed
  estimator and nothing else in the pipeline shifted.
- **Step 2 model-aware column, `d=-0.25`** (`step2_two_regimes.json`):
  `d_hat_typical_agent_model_aware` runs `-0.2481` (`J=0`) to `-0.2467`
  (`J=0.5946`), i.e. still tight and `J`-independent, matching the
  qualitative claim already made in the Sept-8 entry. This column fits the
  **ensemble-averaged** periodogram (`N=400` agents), not a single agent's,
  and the ensemble periodogram was already clean enough for the old 9-start
  grid to find the right basin -- so step 2 was never actually exposed to
  this bug and is unaffected in any material way by the fix. (No exact
  pre-fix numbers were recorded for this column to diff against, since the
  Sept-8 entry only described it qualitatively; re-derivable from
  `step2_two_regimes.json` if ever needed.)
- **Step 1 V4** (`step1_v4_whittle_dj.json`): scope is `d>0` only
  (`PROJECT.md` section 7's V4 scope), so not exposed to this bug; re-run
  for completeness, numbers unchanged in substance.
- **Step 4 N-scaling** (`step4_n_scaling.json`): swept at `d=0.25` only (not
  the buggy sign) -- `field_variance` is bit-for-bit identical to the
  Sept-10 numbers at every `N` (it never calls `whittle_dj_estimate`), so
  item 4a's finding and its "not resolved" status are completely unaffected
  by this fix. `isolated_variance` (single agent, index 0, one fit per
  realization) shifted at `N=50,100` for `J=0.3` (`0.0095/0.00955 ->
  0.01382/0.01393`) but not at `N=400,1600` (`0.00960/0.00963` unchanged) --
  a single realization at small `N` landing in a different basin than
  before, not a systematic pattern; `n_realizations=10` is too few draws to
  read anything into this beyond "the fix touches individual fits
  opportunistically, including occasionally on the `d>0` side," which is
  expected of any multi-start widening and not something this entry
  investigates further (item 4a itself remains out of scope, per the
  Sept-10 entry's own deferral).

**Does this move the headline Phase-3 physics?** No. Step 2's actual
sign-asymmetry result (`PROJECT.md` section 7's coupling-kills-superdiffusive-
memory-not-subdiffusive-memory claim) was established via the **windowed
EA-MSD fit** (`alpha_typical_agent`), which is model-free and was never
touched by this bug -- the Sept-8 entry chose that fit specifically because
the model-aware and naive Whittle variants were both known to be
informative-but-imperfect stand-ins, not the primary evidence. This fix
makes the model-aware *column* more trustworthy (steps 3-4 now show a
smaller, more honest gap between model-aware and model-agnostic for `d<0`,
strong `J`), but no conclusion already drawn in `PROJECT.md` or in NOTES.md's
"Steps 3-5" / Sept-8 / Sept-10 physics discussion rested on the buggy
numbers.

**Verdict.** Item 3 is now fully closed, not closed-with-a-caveat: the
`d>0` side was always a genuine finite-sample identifiability wall (still
true, still 10/60 pinned, unchanged by the wider grid -- this is the correct,
expected residue, not a sign the fix is incomplete); the `d<0` side was a
real optimizer bug and is now fixed, verified both by the previously-red
unit test (now green) and by the residual-surface diagnostic's
`grid_matches_optimizer_result` flipping to `True`. The Sept-10 entry's
recommendation ("widen `whittle_dj_estimate`'s multi-start grid ... before
Phase 4 leans on the isolated model-aware observer for `d<0`, strong-`J`
populations") is done. Item 4a (field-variance-vs-`N`) remains open and
out of scope for this entry, exactly as the Sept-10 verdict left it -- three
of the original four items (1, 2, 3) are now clean; item 4a is the one
remaining flag before Phase 4 should lean on the field-aware observer's
precision at `N != 400`.

