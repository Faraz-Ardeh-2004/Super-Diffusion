"""Static diagnostic figures.

Every function returns its :class:`~matplotlib.figure.Figure` and none calls
``show()``: a notebook decides when to display, and a script decides where to
save. Colour comes from :mod:`collectivediff.viz.palette` and nowhere else.

Each figure carries a legend whenever it draws more than one series, and the
curve plots also label their lines directly, so identity never rests on colour
alone.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..estimators import (
    PowerLawFit,
    amplitude_scatter,
    ea_msd,
    gaussian_reference,
    increment_acf,
    van_hove,
)
from ..estimators.correlations import fbm_increment_acf
from ..estimators.moments import MomentSpectrum, levy_walk_nu
from . import palette

__all__ = [
    "plot_amplitude_scatter",
    "plot_exponent_spread",
    "plot_increment_acf",
    "plot_moment_spectrum",
    "plot_msd",
    "plot_single_trajectory_panel",
    "plot_trajectories",
    "plot_van_hove",
]


def _figure(width: float = 6.4, height: float = 4.2, **kwargs: Any) -> tuple[Figure, Any]:
    """Create a figure and axes under the project style."""
    import matplotlib.pyplot as plt

    with plt.rc_context(palette.rc_params()):
        fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    fig.patch.set_facecolor(palette.SURFACE)
    return fig, ax


def _merge_intervals(
    intervals: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Collapse overlapping ``(low, high)`` pairs into disjoint spans."""
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for low, high in ordered[1:]:
        last_low, last_high = merged[-1]
        if low <= last_high:
            merged[-1] = (last_low, max(last_high, high))
        else:
            merged.append((low, high))
    return merged


def _label_line_end(ax: Any, x: NDArray, y: NDArray, text: str, colour: str) -> None:
    """Write a series name at the right-hand end of its curve.

    Direct labelling is the relief required by three of the palette slots, whose
    contrast against the light surface is below 3:1, and it saves the reader a
    trip to the legend on a crowded log-log plot.
    """
    finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if not finite.any():
        return
    ax.annotate(
        text,
        xy=(x[finite][-1], y[finite][-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=colour,
        fontsize=8,
        va="center",
        ha="left",
        clip_on=False,
    )


def plot_msd(
    curves: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    fits: Mapping[str, PowerLawFit] | None = None,
    reference_slopes: Mapping[str, float] | None = None,
    title: str = "Ensemble MSD",
    xlabel: str = "time $t$",
    ylabel: str = r"$\langle r^2(t)\rangle$",
    fig_ax: tuple[Figure, Any] | None = None,
) -> Figure:
    """Log-log MSD with the fit window shaded and the fitted slope annotated.

    Parameters
    ----------
    curves : mapping
        ``label -> (t, msd)``. The label's leading token is looked up in
        :data:`~collectivediff.viz.palette.MECHANISM_COLOUR`, so
        ``"fbm H=0.7"`` and ``"fbm H=0.3"`` share a hue and differ by line
        style.
    fits : mapping, optional
        ``label -> PowerLawFit``. Each fit's window is shaded and its exponent
        written into the legend entry with the bootstrap error, not the
        least-squares one.
    reference_slopes : mapping, optional
        ``label -> analytic exponent``, drawn as a thin neutral guide line so
        that a deviation is visible as a divergence rather than having to be
        read off the numbers.
    title, xlabel, ylabel : str
        Axis furniture.
    fig_ax : tuple, optional
        Existing ``(figure, axes)`` to draw into.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The shaded window is the point of the figure as much as the curves are:
    ``PROJECT.md`` section 2 forbids fitting the full lag range, and a reader
    who cannot see which decade was fitted cannot judge the exponent.
    """
    fig, ax = fig_ax if fig_ax is not None else _figure(
        width=6.8 if len(curves) <= 4 else 9.0
    )
    styles: dict[str, int] = {}

    # Merge the fit windows into disjoint spans before shading. Drawing each
    # window separately would stack the alpha where they overlap and invent a
    # darker band that means nothing.
    if fits:
        for low, high in _merge_intervals(fit.window for fit in fits.values()):
            ax.axvspan(low, high, color=palette.SHADE, alpha=0.3, lw=0, zorder=0)

    # A legend is always present; direct labels are added only when few enough
    # curves are on the axes for them not to collide.
    direct = len(curves) <= 4

    for label, (t, msd) in curves.items():
        tag = label.split()[0]
        colour = palette.mechanism_colour(tag)
        index = styles.get(tag, 0)
        styles[tag] = index + 1
        style = palette.LINE_STYLES[index % len(palette.LINE_STYLES)]

        legend = palette.mechanism_label(tag) + label[len(tag) :]
        if fits and label in fits:
            fit = fits[label]
            error = fit.bootstrap_err
            legend += (
                rf"  $\alpha$={fit.exponent:.3f}"
                + (rf"$\pm${error:.3f}" if error is not None else "")
            )
        ax.plot(t, msd, color=colour, ls=style, label=legend, zorder=3)
        if direct:
            _label_line_end(
                ax,
                np.asarray(t),
                np.asarray(msd),
                palette.mechanism_label(tag) + label[len(tag) :],
                colour,
            )

    if reference_slopes:
        for label, slope in reference_slopes.items():
            t, msd = curves[label]
            t = np.asarray(t, dtype=np.float64)
            positive = t > 0.0
            anchor_t = t[positive][len(t[positive]) // 4]
            anchor_y = np.asarray(msd)[positive][len(t[positive]) // 4]
            guide = anchor_y * (t[positive] / anchor_t) ** slope
            ax.plot(
                t[positive],
                guide,
                color=palette.REFERENCE,
                lw=1.0,
                ls=(0, (2, 3)),
                zorder=2,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(curves) > 1:
        if direct:
            # Few enough curves that the legend can sit inside without hiding one.
            ax.legend(loc="upper left")
        else:
            # Crowded: the legend goes beside the axes rather than over a curve.
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    if fits:
        ax.text(
            0.98,
            0.03,
            "shaded: fit window",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=palette.TEXT_SECONDARY,
        )
    # Direct labels sit outside the axes with clipping off, and an external
    # legend is not an axes child; tight_layout sees neither, so the margin is
    # reserved afterwards rather than through its rect.
    fig.tight_layout()
    fig.subplots_adjust(right=0.82 if direct else 0.62)
    return fig


def plot_van_hove(
    trajectories: NDArray[Any],
    t_indices: Sequence[int],
    mechanism: Any = "brownian",
    bins: int | NDArray[np.float64] = 61,
    rescale: bool = True,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Displacement distributions at several times, over a Gaussian reference.

    Rescaling each distribution by its own standard deviation is what turns this
    into a *test*: a self-similar Gaussian process collapses every time onto the
    same curve, so both a failure to collapse and a departure from the Gaussian
    are visible at once. Diffusing diffusivity is the case to look at -- it
    collapses badly at short times and approaches the Gaussian at long ones,
    while its MSD says nothing at all.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    t_indices : sequence of int
        Times at which to take the distribution.
    mechanism : str or GeneratorConfig
        Sets the hue; times are distinguished by opacity within that hue.
    bins : int or ndarray
        Passed to :func:`~collectivediff.estimators.distributions.van_hove`.
    rescale : bool
        Divide by the standard deviation before histogramming.
    title : str, optional
        Defaults to the mechanism name.
    ylim : tuple of float, optional
        Explicit y-axis limits ``(ymin, ymax)``. Defaults to ``(1e-4, ymax)``
        scaled to clear the highest peak.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    This figure wants a large ensemble -- 20000 particles or more. Its subject
    is the *tails*, which is where the counts are thinnest: at a few thousand
    particles the outer bins hold single trajectories and the curve dissolves
    into noise exactly where it should be read.
    """
    fig, ax = _figure()
    colour = palette.mechanism_colour(mechanism)
    alphas = np.linspace(0.35, 1.0, len(t_indices))
    max_density = 0.0

    for alpha, t_index in zip(alphas, t_indices):
        centres, density = van_hove(
            trajectories, t_index, bins=bins, rescale=rescale
        )
        pos = density[density > 0.0]
        if pos.size > 0:
            max_density = max(max_density, float(np.nanmax(pos)))
        # An empty bin is not a small probability, it is no measurement at all.
        # Plotting it as zero on a log axis would draw a spike to the bottom of
        # the frame and make the tails -- the whole point of the figure -- look
        # like noise.
        ax.plot(
            centres,
            np.where(density > 0.0, density, np.nan),
            color=colour,
            alpha=float(alpha),
            label=f"$t$ = {t_index}",
            zorder=3,
        )

    if rescale:
        grid = np.linspace(-5.0, 5.0, 401)
        ax.plot(
            grid,
            gaussian_reference(grid),
            color=palette.REFERENCE,
            lw=1.2,
            ls=(0, (2, 3)),
            label="Gaussian",
            zorder=2,
        )
        ax.set_xlabel(r"$\Delta x / \sigma$")
        ax.set_xlim(-5.0, 5.0)
    else:
        ax.set_xlabel(r"$\Delta x$")

    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(1e-4, max(1.0, max_density * 1.4))
    ax.set_ylabel(r"$G_s(\Delta x, t)$")
    ax.set_title(title or f"van Hove distribution: {palette.mechanism_label(mechanism)}")
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    return fig


def plot_amplitude_scatter(
    trajectories: NDArray[Any],
    lag: int,
    mechanism: Any = "brownian",
    cfg: Any | None = None,
    plateau: float | None = None,
    bins: int = 40,
    title: str | None = None,
) -> Figure:
    """Histogram of ``xi = TA-MSD / <TA-MSD>``, the amplitude scatter.

    The figure that makes ergodicity breaking visible rather than numerical. An
    ergodic process piles up on ``xi = 1``; a subdiffusive CTRW spreads across
    the whole axis and keeps a finite density at ``xi = 0`` -- trajectories that
    spent the entire measurement inside one trapping event.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    lag : int
        Lag at which the TA-MSD is evaluated.
    mechanism : str or GeneratorConfig
        Sets the hue.
    cfg : GeneratorConfig, optional
        Passed through to guard divergent moments.
    plateau : float, optional
        Theoretical EB value to quote alongside the measured one.
    bins : int
        Histogram resolution.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _figure()
    xi = amplitude_scatter(trajectories, lag, cfg)
    colour = palette.mechanism_colour(mechanism)

    # One series, so no legend box: the title names it (and a legend here would
    # sit exactly where the EB caption belongs).
    ax.hist(
        xi,
        bins=bins,
        range=(0.0, max(3.0, float(np.percentile(xi, 99)))),
        density=True,
        color=colour,
        alpha=0.8,
    )
    ax.axvline(1.0, color=palette.REFERENCE, lw=1.2, ls=(0, (2, 3)))
    ax.annotate(
        r"$\xi = 1$ (ergodic limit)",
        xy=(1.0, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=palette.TEXT_SECONDARY,
    )

    measured = float(np.mean(xi**2) - 1.0)
    caption = rf"EB = {measured:.3f}"
    if plateau is not None:
        caption += rf"   (theory {plateau:.3f})"
    ax.text(
        0.98,
        0.92,
        caption,
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=palette.TEXT_PRIMARY,
    )

    ax.set_xlabel(r"$\xi = \overline{\delta^2}/\langle\overline{\delta^2}\rangle$")
    ax.set_ylabel("density")
    ax.set_title(
        title
        or f"Amplitude scatter at lag {lag}: {palette.mechanism_label(mechanism)}"
    )
    fig.tight_layout()
    return fig


def plot_increment_acf(
    curves: Mapping[str, NDArray[np.float64]],
    hurst_reference: float | Sequence[float] | None = None,
    title: str = "Increment autocorrelation",
) -> Figure:
    """Increment ACF, with the fBm closed form overlaid when relevant.

    The observable that says *why* a process is anomalous. The zero line is
    drawn heavily because its two sides are different physics: below it the
    increments reverse, above it they persist.

    Parameters
    ----------
    curves : mapping
        ``label -> acf array`` indexed from lag 0.
    hurst_reference : float or sequence of float, optional
        If given, overlay ``rho(k)`` for fBm at this (or these) Hurst exponent(s).
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _figure()
    styles: dict[str, int] = {}

    for label, acf in curves.items():
        tag = label.split()[0]
        colour = palette.mechanism_colour(tag)
        index = styles.get(tag, 0)
        styles[tag] = index + 1
        lags = np.arange(len(acf))
        ax.plot(
            lags,
            acf,
            color=colour,
            ls=palette.LINE_STYLES[index % len(palette.LINE_STYLES)],
            marker="o",
            ms=4,
            label=palette.mechanism_label(tag) + label[len(tag) :],
            zorder=3,
        )

    if hurst_reference is not None:
        refs = (
            [hurst_reference]
            if isinstance(hurst_reference, (int, float))
            else list(hurst_reference)
        )
        lags = np.arange(max(len(a) for a in curves.values()))
        ref_styles = [(0, (2, 3)), (0, (4, 2)), (0, (3, 1, 1, 1))]
        for i, h in enumerate(refs):
            ax.plot(
                lags,
                fbm_increment_acf(lags, float(h)),
                color=palette.REFERENCE,
                lw=1.2,
                ls=ref_styles[i % len(ref_styles)],
                label=rf"fBm theory, $H$={h}",
                zorder=2,
            )

    ax.axhline(0.0, color=palette.TEXT_SECONDARY, lw=1.0, zorder=1)
    ax.set_xlabel("lag $k$")
    ax.set_ylabel(r"$\rho(k)$")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_moment_spectrum(
    spectra: Mapping[str, MomentSpectrum],
    gamma_reference: float | None = None,
    title: str = "Moment scaling spectrum",
) -> Figure:
    """``nu(q)`` with a straight-line reference, so a kink is unmistakable.

    A flat spectrum means simple scaling -- one length growing as ``t^nu``. A
    kink means strong anomalous diffusion, and is the only unambiguous signature
    of a Levy walk. The horizontal guide at each spectrum's value at small ``q``
    is what makes the departure legible: without it, a gently rising curve and a
    genuinely bilinear one look alike.

    Parameters
    ----------
    spectra : mapping
        ``label -> MomentSpectrum``.
    gamma_reference : float, optional
        Overlay the analytic Levy-walk form at this ``g`` and mark the kink at
        ``q = g``.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _figure()
    styles: dict[str, int] = {}

    for label, spectrum in spectra.items():
        tag = label.split()[0]
        colour = palette.mechanism_colour(tag)
        index = styles.get(tag, 0)
        styles[tag] = index + 1
        style = palette.LINE_STYLES[index % len(palette.LINE_STYLES)]

        finite = np.isfinite(spectrum.nu_err)
        if finite.any():
            ax.errorbar(
                spectrum.q,
                spectrum.nu,
                yerr=np.where(finite, spectrum.nu_err, 0.0),
                color=colour,
                ls=style,
                marker="o",
                ms=4,
                capsize=2,
                label=palette.mechanism_label(tag) + label[len(tag) :],
                zorder=3,
            )
        else:
            ax.plot(
                spectrum.q,
                spectrum.nu,
                color=colour,
                ls=style,
                marker="o",
                ms=4,
                label=palette.mechanism_label(tag) + label[len(tag) :],
                zorder=3,
            )
        # The straight-line reference: what simple scaling would look like.
        # Anchored on the median rather than on nu(q_min), which is the noisiest
        # point of the spectrum and would tilt the comparison.
        ax.axhline(
            float(np.nanmedian(spectrum.nu)),
            color=colour,
            lw=0.9,
            ls=(0, (1, 4)),
            zorder=1,
        )

    if gamma_reference is not None:
        q_grid = np.linspace(
            min(s.q.min() for s in spectra.values()),
            max(s.q.max() for s in spectra.values()),
            300,
        )
        ax.plot(
            q_grid,
            levy_walk_nu(q_grid, gamma_reference),
            color=palette.REFERENCE,
            lw=1.2,
            ls=(0, (2, 3)),
            label=rf"Levy-walk theory, $g$={gamma_reference}",
            zorder=2,
        )
        ax.axvline(gamma_reference, color=palette.REFERENCE, lw=0.9, alpha=0.6)
        # Annotated at the top: the bottom-left corner carries the caption, and
        # the two collide when the spectrum runs low.
        ax.annotate(
            rf"kink at $q=g={gamma_reference}$",
            xy=(gamma_reference, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            va="top",
            color=palette.TEXT_SECONDARY,
        )

    ax.set_xlabel("moment order $q$")
    ax.set_ylabel(r"$\nu(q)$")
    ax.set_title(title)
    ax.text(
        0.02,
        0.03,
        "dotted horizontal: simple-scaling reference",
        transform=ax.transAxes,
        fontsize=8,
        color=palette.TEXT_SECONDARY,
    )
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_exponent_spread(
    spreads: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    truths: Mapping[str, float],
    title: str = "Single-trajectory exponent estimate",
    xlabel: str = "trajectory length $T$",
    ylabel: str = r"fitted $\alpha$ per trajectory",
    overlays: Mapping[str, tuple[NDArray[np.float64], str]] | None = None,
) -> Figure:
    """Spread of the single-trajectory exponent against trajectory length.

    The estimator-variance study of ``PHASE1_PROMPT.md`` step 5, and the figure
    that carries the phase-1 conclusion: for an ergodic mechanism the cloud of
    per-trajectory estimates narrows onto the true exponent as ``T`` grows,
    while for a CTRW it does not narrow at all and is not even centred on the
    right value.

    Drawn as an interquartile band plus the median, with the true exponent as a
    dashed rule -- percentiles rather than mean and standard deviation, because
    the CTRW distribution is broad and skewed and a standard deviation would
    flatter it.

    Parameters
    ----------
    spreads : mapping
        ``label -> (lengths, estimates)`` where ``estimates`` has shape
        ``(n_lengths, n_trajectories)``.
    truths : mapping
        ``label -> true exponent``.
    title, xlabel, ylabel : str
    overlays : mapping, optional
        ``label -> (estimates, overlay_label)`` for companion estimates
        to overlay on the same subplot (e.g. corrected Phase 2 estimates).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(spreads)
    import matplotlib.pyplot as plt

    with plt.rc_context(palette.rc_params()):
        fig, axes = plt.subplots(
            1, n, figsize=(3.0 * n, 3.4), sharey=True, squeeze=False
        )
    fig.patch.set_facecolor(palette.SURFACE)

    for column, (ax, (label, (lengths, estimates))) in enumerate(
        zip(axes[0], spreads.items())
    ):
        tag = label.split()[0]
        colour = palette.mechanism_colour(tag)
        low, mid, high = np.nanpercentile(estimates, [25, 50, 75], axis=1)

        has_overlay = overlays is not None and label in overlays
        naive_label = "naive OLS" if has_overlay else "interquartile"
        median_label = "naive median" if has_overlay else "median"

        ax.fill_between(
            lengths, low, high, color=colour, alpha=0.30, lw=0, label=naive_label
        )
        ax.plot(lengths, mid, color=colour, marker="o", ms=5, label=median_label)

        if has_overlay:
            corr_estimates, corr_label = overlays[label]
            c_low, c_mid, c_high = np.nanpercentile(corr_estimates, [25, 50, 75], axis=1)
            ax.fill_between(
                lengths,
                c_low,
                c_high,
                color=palette.TEXT_PRIMARY,
                alpha=0.15,
                lw=1.0,
                ls="--",
                edgecolor=palette.TEXT_PRIMARY,
                label=f"{corr_label} IQR",
            )
            ax.plot(
                lengths,
                c_mid,
                color=palette.TEXT_PRIMARY,
                marker="s",
                ms=4,
                ls="--",
                lw=1.3,
                label=f"{corr_label} median",
            )

        ax.axhline(
            truths[label],
            color=palette.REFERENCE,
            lw=1.2,
            ls=(0, (2, 3)),
            label="true exponent",
        )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(palette.mechanism_label(tag) + label[len(tag) :])
        # One legend for the panel: every subplot uses the same three encodings,
        # so repeating it would only cover data.
        if column == 0 and not has_overlay:
            ax.legend(loc="lower right", fontsize=8)
        elif has_overlay:
            ax.legend(loc="lower left" if np.nanmedian(mid) > 1.2 else "upper right", fontsize=7.5)

    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_single_trajectory_panel(
    report: Mapping[str, Mapping[str, Any]],
    title: str = "What one trajectory says about itself",
) -> Figure:
    """Per mechanism: the raw trace above, its own TA-MSD and fit below.

    The figure that carries ``01_single_particle.ipynb``. The top row shows
    traces that a reader can tell apart by eye -- the flat stretches of a CTRW,
    the straight runs of a Levy walk. The bottom row shows what the standard
    single-trajectory estimator makes of each, with the true ensemble exponent
    marked, and the answers are far less distinguishable than the traces.

    The CTRW column is the one to read: its fitted exponent lands near 1
    whatever the ensemble exponent is, because the time-averaged MSD of a single
    CTRW trajectory is linear in the lag. That is not noise, and no longer
    observation removes it.

    Parameters
    ----------
    report : mapping
        Output of
        :func:`collectivediff.studies.single_trajectory_report`.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n = len(report)
    with plt.rc_context(palette.rc_params()):
        fig, axes = plt.subplots(
            2, n, figsize=(2.5 * n, 5.4), squeeze=False, height_ratios=(1.0, 1.3)
        )
    fig.patch.set_facecolor(palette.SURFACE)

    for column, (label, entry) in enumerate(report.items()):
        tag = label.split()[0]
        colour = palette.mechanism_colour(tag)
        name = palette.mechanism_label(tag) + label[len(tag) :]

        trace_ax = axes[0][column]
        trajectory = np.asarray(entry["trajectory"])
        trace_ax.plot(
            np.arange(trajectory.shape[1]), trajectory[0, :, 0], color=colour, lw=1.0
        )
        trace_ax.axhline(0.0, color=palette.TEXT_SECONDARY, lw=0.7)
        trace_ax.set_title(name, fontsize=10)
        trace_ax.set_xlabel("$t$", fontsize=9)
        if column == 0:
            trace_ax.set_ylabel("$x(t)$")

        fit_ax = axes[1][column]
        lags = np.asarray(entry["lags"], dtype=np.float64)
        curve = np.asarray(entry["ta_msd"], dtype=np.float64)
        exponent, truth = float(entry["exponent"]), float(entry["truth"])

        fit_ax.plot(lags, curve, color=colour, marker="o", ms=3, lw=1.4)
        if np.isfinite(exponent):
            anchor = curve[len(curve) // 2] / lags[len(lags) // 2] ** exponent
            fit_ax.plot(
                lags,
                anchor * lags**exponent,
                color=palette.REFERENCE,
                lw=1.0,
                ls=(0, (2, 3)),
            )
        fit_ax.set_xscale("log")
        fit_ax.set_yscale("log")
        fit_ax.set_xlabel(r"lag $\Delta$", fontsize=9)
        if column == 0:
            fit_ax.set_ylabel(r"$\overline{\delta^2(\Delta)}$")

        measured = "no motion" if not np.isfinite(exponent) else f"{exponent:.2f}"
        fit_ax.text(
            0.04,
            0.96,
            f"fitted {measured}\ntrue {truth:.2f}",
            transform=fit_ax.transAxes,
            va="top",
            fontsize=8.5,
            color=palette.TEXT_PRIMARY,
        )

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def plot_trajectories(
    trajectories: NDArray[Any],
    mechanism: Any = "brownian",
    n_show: int = 6,
    title: str | None = None,
) -> Figure:
    """Static traces: position against time in 1-D, or the plane in 2-D.

    The figure a reader should see before any statistic, because several of the
    phase-1 conclusions are visible in the raw traces -- the flat stretches of a
    CTRW, the straight runs of a Levy walk, the reversals of antipersistent fBm.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    mechanism : str or GeneratorConfig
        Sets the hue.
    n_show : int
        How many trajectories to draw.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _figure()
    colour = palette.mechanism_colour(mechanism)
    n_show = min(n_show, trajectories.shape[0])
    alphas = np.linspace(0.45, 1.0, n_show)
    n_dim = trajectories.shape[2]

    for index in range(n_show):
        alpha = float(alphas[index])
        if n_dim == 1:
            ax.plot(
                np.arange(trajectories.shape[1]),
                trajectories[index, :, 0],
                color=colour,
                alpha=alpha,
                lw=1.2,
            )
        else:
            ax.plot(
                trajectories[index, :, 0],
                trajectories[index, :, 1],
                color=colour,
                alpha=alpha,
                lw=1.0,
            )

    if n_dim == 1:
        ax.set_xlabel("time $t$")
        ax.set_ylabel("$x(t)$")
        ax.axhline(0.0, color=palette.TEXT_SECONDARY, lw=0.8)
    else:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.plot([0.0], [0.0], marker="o", ms=5, color=palette.REFERENCE)

    label = palette.mechanism_label(mechanism)
    ax.set_title(title or f"{label}: {n_show} trajectories")
    fig.tight_layout()
    return fig
