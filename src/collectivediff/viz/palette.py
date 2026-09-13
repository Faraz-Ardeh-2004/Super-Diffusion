"""The single place colour is defined.

``PHASE1_PROMPT.md`` step 4 asks for one consistent, colourblind-safe palette.
Two rules make that more than a wish:

* **Colour follows the entity, not its rank.** A mechanism keeps its hue whatever
  else is on the axes, so a reader who has learned "orange is fBm" on one figure
  is not retrained by the next. :data:`MECHANISM_COLOUR` is that fixed mapping,
  and nothing outside this module chooses a colour.
* **Identity is never carried by colour alone.** Every figure function in
  :mod:`collectivediff.viz.static` draws a legend, and the curve plots also
  direct-label their lines.

The eight hues are the validated categorical order of the project's reference
palette. On the seven slots used here the worst adjacent pair separations are
``dE = 9.1`` under protanopia and ``19.6`` for normal vision -- above the ``8``
and ``15`` floors respectively. Three of the slots (aqua, yellow, magenta) fall
below a 3:1 contrast ratio against the light surface, which is why the legend
and direct labels above are a requirement rather than a courtesy.

Distinguishing two parameter values of the *same* mechanism -- fBm at
``H = 0.3`` and ``H = 0.7``, say -- is done with line style, never by promoting
one of them to a different hue.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "CATEGORICAL",
    "GRID",
    "LINE_STYLES",
    "MECHANISM_COLOUR",
    "MECHANISM_LABEL",
    "REFERENCE",
    "SHADE",
    "SURFACE",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "mechanism_colour",
    "mechanism_label",
    "rc_params",
]

#: Chart surface and ink. Painted explicitly so a figure rendered inside a dark
#: notebook theme does not inherit an unreadable transparent background.
SURFACE: Final[str] = "#fcfcfb"
TEXT_PRIMARY: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
GRID: Final[str] = "#dcdbd6"

#: Neutral used for analytic reference curves, so that "theory" is never mistaken
#: for one more mechanism.
REFERENCE: Final[str] = "#52514e"

#: Fill for the shaded fit window.
SHADE: Final[str] = "#c9c8c2"

#: Validated categorical order. Do not reorder: the separation guarantees were
#: measured on adjacent pairs in exactly this sequence.
CATEGORICAL: Final[tuple[str, ...]] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Fixed mechanism -> hue assignment. Brownian takes slot 1 because it is the
#: null hypothesis every other curve is read against.
MECHANISM_COLOUR: Final[dict[str, str]] = {
    "brownian": CATEGORICAL[0],
    "fbm": CATEGORICAL[1],
    "sbm": CATEGORICAL[2],
    "ctrw": CATEGORICAL[3],
    "levy_flight": CATEGORICAL[4],
    "levy_walk": CATEGORICAL[5],
    "ddm": CATEGORICAL[6],
}

#: Human-readable names for legends and titles.
MECHANISM_LABEL: Final[dict[str, str]] = {
    "brownian": "Brownian",
    "fbm": "fBm",
    "sbm": "scaled Brownian",
    "ctrw": "CTRW",
    "levy_flight": "Levy flight",
    "levy_walk": "Levy walk",
    "ddm": "diffusing diffusivity",
}

#: Line styles for distinguishing parameter values within one mechanism.
LINE_STYLES: Final[tuple[str, ...]] = ("-", "--", ":", "-.")


def _tag(mechanism: Any) -> str:
    """Accept either a config object or a mechanism name."""
    return mechanism if isinstance(mechanism, str) else type(mechanism).name


def mechanism_colour(mechanism: Any) -> str:
    """Colour for a mechanism, by name or by config.

    Parameters
    ----------
    mechanism : str or GeneratorConfig
        Mechanism tag such as ``"ctrw"``, or any generator config.

    Returns
    -------
    str
        Hex colour.

    Raises
    ------
    KeyError
        For an unknown mechanism. Failing loudly is deliberate: silently falling
        back to a default hue would break the one-colour-per-entity rule exactly
        when a new mechanism is added.
    """
    tag = _tag(mechanism)
    try:
        return MECHANISM_COLOUR[tag]
    except KeyError as exc:
        raise KeyError(
            f"no colour assigned to {tag!r}; add it to MECHANISM_COLOUR rather "
            f"than choosing a colour at the call site. Known: "
            f"{sorted(MECHANISM_COLOUR)}"
        ) from exc


def mechanism_label(mechanism: Any) -> str:
    """Human-readable label for a mechanism, by name or by config."""
    tag = _tag(mechanism)
    return MECHANISM_LABEL.get(tag, tag)


def rc_params() -> dict[str, Any]:
    """Matplotlib settings shared by every figure in the project.

    Recessive grid and axes, explicit surface colours, and the categorical order
    as the default property cycle so that an ad-hoc plot in a notebook still
    lands inside the palette.

    Returns
    -------
    dict
        Suitable for ``matplotlib.rcParams.update`` or ``plt.rc_context``.
    """
    from cycler import cycler

    return {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
    }
