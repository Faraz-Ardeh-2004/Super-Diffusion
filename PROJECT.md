# PROJECT.md — Inferring collective diffusion regimes from partial agent observation

This file is the durable context for the whole project. Read it before any task.
Do not modify it without being asked.

---

## 1. Research question

Given a population of agents whose microscopic motion rule is unknown, and given
access to only a subset of them, can we identify the **diffusion regime of the
whole population** (subdiffusive / normal / superdiffusive) from:

- the observed agent's own trajectory history (its memory), and
- the instantaneous mean field it experiences?

Sub-questions, in the order they are attacked:

- **Q1 — Estimator quality.** For each generative mechanism, how well can the
  anomalous exponent be recovered from a *single* finite trajectory? Where does
  a single trajectory fail *in principle* (ergodicity breaking) rather than
  merely from finite-sample noise?
- **Q2 — Mechanism discrimination.** Which observables separate mechanisms that
  share the same exponent (e.g. fBm vs CTRW vs Lévy walk)?
- **Q3 — Single agent vs collective.** Once agents are coupled through a mean
  field, how correlated is a single agent's inferred regime with the ensemble's
  true regime? Does adding the mean-field signal as an extra observable help?
- **Q4 — Sampling cost under heterogeneity.** For a population made of `M`
  modules with different exponents, how many observed agents `n` are needed to
  recover the population's exponent mixture? Is the threshold `~M`, `~M log M`,
  or something else, and how does it depend on the sampling strategy?

The working hypothesis from the group discussion is that `n >= M` is *necessary*
but far from sufficient, and that the coupon-collector scale `M log M` is closer
to the truth for uniform random sampling.

---

## 2. Non-negotiable conventions

**Validation first.** A generator is not usable until a test asserts that it
reproduces its analytic ensemble MSD within a stated tolerance, over a stated
time window. No analysis is run on an unvalidated generator. This rule has the
highest priority in the project.

**Determinism.** Every stochastic function takes an explicit
`rng: np.random.Generator`. No calls to global `np.random.*` anywhere. Every
figure is reproducible from `(config, seed)`.

**Config as data.** Simulation parameters live in frozen dataclasses, not in
function-call sites and not in notebooks. Configs are serialisable to JSON and
stored alongside any cached output.

**Package, not notebook.** All logic lives in the package. Notebooks import,
call, and plot — they never define simulation or estimation logic. If a notebook
cell grows past ~15 lines of logic, that logic belongs in the package.

**Caching.** Trajectory arrays are cached to `data/cache/<hash>.npz` where the
hash is derived from the config JSON. Cache is disposable; never commit it.

**Units.** Lattice-free, continuous space. Time is in integer steps of size
`dt = 1` unless a generator intrinsically needs continuous time (CTRW, Lévy
walk), in which case the raw event sequence is generated first and then sampled
onto a uniform grid by an explicit, tested resampling function. Do not blur these
two representations.

**Array layout.** Trajectories are always arrays of shape
`(n_particles, n_steps, n_dim)`. `n_dim` is 1 or 2. No other layout is allowed
to escape a function. Cached arrays are `float32`; estimator accumulations are
`float64`.

**Dimensionality.** All statistics and estimators run in 1D. Two-dimensional
generators exist for visualisation only. Brownian, fBm and scaled Brownian
motion are separable and their 2D forms are thin wrappers over 1D. CTRW and
diffusing diffusivity share a scalar state across components and are not two
independent copies. Lévy flight and Lévy walk are not separable at all and need
explicit isotropic 2D implementations.

**Fitting.** Power-law exponents are fitted by least squares on `log(lag)` or
`log(t)` versus `log(MSD)`. `fit_powerlaw` has **no default window**; every
caller states its own, because the two cases are not the same claim:

- *TA-MSD versus lag* — restrict to `[lag_min, T/10]`. Beyond that the number of
  contributing windows collapses and the estimate is noise.
- *EA-MSD versus time* — every particle contributes at every `t`, so the window
  is instead an assertion about where the asymptotics hold. For CTRW and the
  Lévy walk this window must exclude the early-time regime where finite-time
  corrections dominate.

**Uncertainty is estimated by bootstrap over particles, never from the fit
residuals.** EA-MSD points at different times are built from the same particles
and are strongly correlated, so ordinary least-squares standard errors
underestimate the true exponent uncertainty by more than an order of magnitude
(measured: ~0.0005 reported against ~0.012 seed-to-seed). Resampling particles
with replacement yields both a correct confidence interval and the empirical
lag-lag covariance, which supplies the weights for a generalised least-squares
fit. That weighting matters wherever estimator variance grows with time — most
sharply for the Lévy walk, where the relative error of the MSD grows as
`t^((g-1)/2)` because the fourth moment `~ t^(5-g)` outgrows the square of the
second. `PowerLawFit.exponent_err` is a goodness-of-fit diagnostic only and must
be documented as such.

---

## 3. Repository layout

```
.
├── PROJECT.md
├── pyproject.toml
├── src/collectivediff/
│   ├── __init__.py
│   ├── config.py            # frozen dataclasses for every simulation
│   ├── generators/
│   │   ├── __init__.py      # registry: name -> generator callable
│   │   ├── brownian.py
│   │   ├── fbm.py           # Davies–Harte / Hosking circulant embedding
│   │   ├── ctrw.py          # heavy-tailed waiting times
│   │   ├── levy.py          # Lévy flight and Lévy walk
│   │   ├── sbm.py           # scaled Brownian motion
│   │   └── ddm.py           # diffusing diffusivity
│   ├── estimators/
│   │   ├── msd.py           # EA-MSD, TA-MSD, EA-TA-MSD, power-law fits
│   │   ├── ergodicity.py    # EB parameter, amplitude scatter distribution
│   │   ├── distributions.py # van Hove, non-Gaussian parameter
│   │   ├── correlations.py  # increment/velocity autocorrelation
│   │   ├── moments.py       # moment scaling spectrum nu(q)
│   │   └── firstpassage.py  # first-passage and return statistics
│   ├── dynamics/            # phase 3+: coupled agents
│   │   ├── meanfield.py
│   │   └── modular.py
│   ├── inference/           # phase 3+: single-agent -> ensemble regime
│   ├── sampling/            # phase 4: which agents to observe
│   ├── viz/
│   │   ├── static.py
│   │   └── animate.py
│   └── validation/          # analytic reference MSDs + tolerance checks
├── notebooks/
│   ├── 01_single_particle.ipynb
│   ├── 02_ensemble.ipynb
│   ├── 03_meanfield.ipynb
│   └── 04_modular.ipynb
├── tests/
└── data/cache/              # gitignored
```

---

## 4. Generative mechanisms and their analytic signatures

All exponents below refer to the ensemble MSD scaling
`<x^2(t)> ~ t^alpha`.

| Mechanism | Regime | alpha | Ergodic | Gaussian propagator | Increment ACF |
|---|---|---|---|---|---|
| Brownian | normal | 1 | yes | yes | delta |
| fBm, H < 1/2 | sub | 2H | yes | yes | negative |
| fBm, H > 1/2 | super | 2H | yes | yes | positive |
| CTRW, waiting exponent `a` in (0,1) | sub | a | **no** | no | zero + long pauses |
| Lévy walk, `g` in (1,2) | super | 3 - g | **no** | no | ballistic stretches |
| Lévy flight, stability `s` in (0,2) | super | MSD diverges | **no** | no (stable law) | delta |
| Scaled Brownian, `a` | either | a | **no** | yes | delta, non-stationary |
| Diffusing diffusivity | normal | 1 | yes | **no** (exponential tails) | delta |

Two entries deserve emphasis because they are the crux of Q1 and Q2:

**CTRW.** Weak ergodicity breaking. The time-averaged MSD of a single trajectory
scales *linearly* in the lag regardless of `a`, and its amplitude remains a
random variable that does not converge as the trajectory lengthens. A single
trajectory therefore cannot reveal the exponent, no matter how long it is or how
much memory the observer has. This is a hard information-theoretic wall and
should be demonstrated explicitly, not merely asserted.
Reference: He, Burov, Metzler, Barkai, PRL 101, 058101 (2008).

Jump lengths are Gaussian — finite variance, pure waiting-time subdiffusion with
no Lévy statistics mixed in. This is deliberate: the scientific point of Q2 is to
contrast CTRW against fBm at `H < 1/2`, which shares its exponent but not its
ergodicity. Heavy-tailed jumps would blur that contrast into the Lévy walk.

CTRW carries an explicit **aging time** `t_a` in its config, defaulting to zero
(ordinary, non-aged). Aging is not a side detail here: an observer who begins
watching an agent at an arbitrary moment is by construction observing an aged
CTRW, and aging changes the ensemble MSD, the time-averaged MSD and the EB
plateau. Phase 1 validates the non-aged case; the parameter exists from the start
so that phases 3 and 4 do not require re-validation.

The mean renewal count has a finite-time correction that makes the fitted
exponent biased, with a sign that reverses as `a` increases:

```
<n(t)> ~= (t/tau0)^a / (Gamma(1-a) Gamma(1+a))
         * [1 + a/((1-a) Gamma(1-a)) * (tau0/t)^(1-a)]  -  1
```

The additive `-1` dominates at small `a` and biases the exponent upward; the
slowly decaying `(tau0/t)^(1-a)` term dominates as `a -> 1` and biases it
downward. The zero crossing sits near `a ~ 0.55-0.6` for a fit window of
`[0.1T, T]`. Because this is known in closed form it must be *corrected for*,
not merely avoided by excluding extreme `a` from tests — an uncorrected,
`a`-dependent bias would contaminate the phase 2 mechanism-classification results
with an artefact we already understand analytically. Implement the finite-time
form as a reference function and either fit it directly or report the predicted
bias alongside the pure power-law fit.

**Lévy flight.** The second moment diverges, so MSD is meaningless. Use
fractional moments of order `q < s`, or the median absolute displacement, or
the growth of the interquantile range. Any code path that computes an MSD for a
Lévy flight must raise or warn rather than silently return a finite number from
a diverging quantity.

---

## 5. Estimator definitions

Time-averaged mean squared displacement of a single trajectory of length `T`:

```
TA-MSD(D) = mean over t in [0, T-D] of  (x(t+D) - x(t))^2
```

Ensemble-averaged TA-MSD is the average of the above over particles.

Ergodicity-breaking parameter, with `xi = TA-MSD / <TA-MSD>`:

```
EB(D) = Var(xi) = <xi^2> - 1
```

For an ergodic process `EB -> 0` as `T/D -> infinity`. For CTRW it converges to
a nonzero constant depending only on the waiting-time exponent. Reproducing that
plateau is a required validation check.

Non-Gaussian parameter in `d` dimensions:

```
a2(t) = <r^4> / ((1 + 2/d) * <r^2>^2) - 1
```

Moment scaling spectrum: fit `<|x|^q> ~ t^(q * nu(q))` for `q` on a grid, e.g.
`q in [0.2, 4]`. `nu(q)` constant means simple scaling; a piecewise-linear
`nu(q)` with a kink means strong anomalous diffusion and is the fingerprint of
Lévy walks.

---

## 6. Phase roadmap

- **Phase 1 — Single and ensemble, no interaction.** Generators, validation,
  estimator toolbox, animations, and the estimator-variance study. Answers Q1
  and Q2. This is where the project's foundations are either solid or rotten.
- **Phase 2 — Estimator identifiability.** Systematic study: given trajectory
  length `T` and ensemble size `N`, what is the bias and variance of each
  estimator per mechanism? Produce a confusion matrix for mechanism
  classification from single trajectories.
- **Phase 3 — Mean-field coupling.** Introduce the coupled dynamics below,
  measure the correlation between the regime inferred from one agent and the
  true ensemble regime, and test whether feeding the local mean-field history as
  an extra feature improves that inference. Answers Q3.
- **Phase 4 — Modular heterogeneity.** `M` modules with distinct exponents and
  block-structured coupling. Module labels are hidden from the estimator. Sweep
  the number of observed agents `n` and the sampling strategy. Answers Q4.

Phases are gated: phase `k+1` does not start until phase `k`'s tests pass and
its figures are reproducible from a clean cache.

---

## 7. Coupled dynamics (phase 3 specification)

Revised before Phase 3 began. Two changes from the original sketch, both
forced by the validation-first rule.

**Memory kernel: ARFIMA weights, not an arbitrary normalised power law.** The
velocity of each agent follows the AR(∞) representation of `ARFIMA(0, d, 0)`,
truncated at `K` lags:

```
v_i(t+1) = sum_{s=1..K} kappa(s) * v_i(t+1-s)
         + J * ( <v>(t) - v_i(t) )
         + sigma * eta_i(t+1)

kappa(s) = -c_s,   where (1 - B)^d = sum_{s>=0} c_s B^s
         = Gamma(s - d) / ( Gamma(-d) Gamma(s + 1) ),   s >= 1
```

`kappa(s)` decays as `s^(-1-d)` — power-law, as before — and the `J = 0` limit
is *exactly* `ARFIMA(0, d, 0)` velocity (up to the `K` truncation), **not**
fractional Gaussian noise. The two share the same asymptotic exponent but not
the same short-lag correlation: `ARFIMA`'s closed form is
`rho(k) = Gamma(1-d) Gamma(k+d) / (Gamma(d) Gamma(k+1-d))`, giving
`rho(1) = d / (1-d)`, against fGn's `rho(1) = 2^(2d) - 1` — different numbers
for any `d != 0`. (An earlier draft of this section claimed the `J = 0` limit
was exactly fGn; it isn't, and a Phase 3 validation built on that claim failed
until this was corrected — the two processes agree asymptotically, which is
enough for the alpha below, but not at short lag.)

```
alpha = 1 + 2d,     d in (-1/2, 1/2)
```

is the ARFIMA/fBm-shared asymptotic exponent — one parameter covers sub-,
normal, and superdiffusion. Convergence to this asymptote from short lag is
fast (within a few percent by lag 10 for `|d|` up to 0.45), but a fit window
`[lag_min, K/10]` should still be checked against the exact ARFIMA prediction
for that window, not asserted equal to `1 + 2d`. Memory is exact for lags
`<= K`; every claim about the coupled model is restricted to lags `<= K/10`.

**Coupling: consensus form, not additive.** `J (<v> - v_i)`, not `+J <v>`.
With `sum_s kappa(s) = 1` (a property of the ARFIMA weights), the additive form
makes the collective mode unstable for any `J > 0`. The consensus form is the
Vicsek/Kuramoto form and has the consequence that drives the whole phase:

**Conservation law.** For all-to-all coupling the `J` term cancels exactly in
the population average, so the mean field `<v>(t)` is `ARFIMA(0, d, 0)` with
variance `sigma^2 / N` **for every value of `J`**. Coupling cannot change the
regime of the coherent mode. This is an exact validation target: Whittle on
`<v>` must recover `d` at the fGn Cramér–Rao efficiency from Phase 2,
independent of `J`.

**Deviation spectrum.** With `delta_i = v_i - <v>`:

```
S_delta(omega) = sigma^2 / | (1 - e^{-i omega})^d + J e^{-i omega} |^2
```

closed form, testable against the periodogram. At low frequency the behaviour
depends on the **sign of `d`**, and the two cases are qualitatively different:

- **`d > 0` (superdiffusive memory).** `(i omega)^d -> 0`, so the denominator
  tends to `J` and `S_delta(0) = sigma^2 / J^2` is finite: the deviation is
  normally diffusive at long times. Memory survives only below a crossover

  ```
  tau_c ~ J^(-1/|d|)
  ```

- **`d < 0` (subdiffusive memory).** `(i omega)^d -> infinity`, so the
  fractional term dominates `J` at every reachable frequency and
  `S_delta(omega) ~ omega^(2|d|) -> 0`. Coupling does **not** destroy the
  anomalous exponent, and `tau_c` has no meaning. Verified numerically: at
  `d = -0.25` the deviation exponent stays near 0.5 for `J` up to 0.5, while at
  `d = +0.25` it falls from 1.48 to 1.19 over the same range.

This asymmetry is a result, not a technicality: **coupling kills superdiffusive
memory but not subdiffusive memory.** Whether a single-agent observer can reach
the collective regime therefore depends on the sign of `d`. V4 and the
crossover study apply to `d > 0` only; for `d < 0` the corresponding test is
that the exponent is *invariant* under `J`.

**Predicted picture, to be tested not assumed.** A single agent's local
exponent runs `1 + 2d` for lags below `tau_c` and `1` above. The coherent mode
retains `1 + 2d` at all times but at amplitude `1/N`, so within reachable `T`
the ensemble MSD reads as normal. Hence the two definitions of "collective
regime" diverge and both must be reported:

- **typical-agent regime** — exponent of the ensemble EA-MSD (≈ 1 for `J > 0`
  in reach);
- **coherent regime** — exponent of the center-of-mass / mean-field mode
  (`1 + 2d` always).

Q3 is closed on the coherent regime. The isolated observer reaches it only
through the agent's own short-lag memory (below `tau_c`); the field-aware
observer reaches it directly through `<v>`. That is the operational content of
"own memory plus mean field" from the original brief.

Vicsek-type ordering is deliberately excluded: the consensus form conserves the
mean and has no flocking transition. Sparse networks and heterogeneous `d_k`
are Phase 4.

---

## 8. Key references

- Metzler & Klafter, Phys. Rep. 339, 1 (2000) — anomalous diffusion review.
- Metzler, Jeon, Cherstvy, Barkai, PCCP 16, 24128 (2014) — ergodicity breaking,
  single-particle tracking, and which observable to use when. The most directly
  relevant reference for phases 1 and 2.
- He, Burov, Metzler, Barkai, PRL 101, 058101 (2008) — weak ergodicity breaking.
- Chechkin, Seno, Metzler, Sokolov, PRX 7, 021002 (2017) — diffusing
  diffusivity, Brownian yet non-Gaussian.
- Zaburdaev, Denisov, Klafter, RMP 87, 483 (2015) — Lévy walks.
- Castiglione, Mazzino, Muratore-Ginanneschi, Vulpiani, Physica D 134, 75 (1999)
  — strong anomalous diffusion and the moment spectrum.
- Liu, Slotine, Barabási, PNAS 110, 2460 (2013) — observability of complex
  networks; the natural framing for the `n` vs `M` question in phase 4.
