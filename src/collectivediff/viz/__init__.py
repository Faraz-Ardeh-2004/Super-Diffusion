"""Static figures and animations.

Every figure function returns its figure object; nothing here calls ``show()``.
Colour is defined once, in :mod:`collectivediff.viz.palette`, and chosen nowhere
else.

Figures are created through ``pyplot`` -- animations need its canvas -- which
means they stay in its global registry until closed. A notebook cell that
displays one figure is fine; a *loop* over mechanisms should call
``matplotlib.pyplot.close(fig)`` once it is done with each, or matplotlib will
hold every one of them in memory and eventually say so.
"""

from __future__ import annotations

from . import palette
from .animate import (
    STYLES,
    AnimationStyle,
    animate_trajectories,
    display_animation,
    save_animation,
)
from .identifiability import (
    plot_conditional_exponent_comparison,
    plot_confusion_matrix_family,
    plot_feature_importance_ranking,
    plot_information_floor_efficiency,
    plot_pairwise_error_vs_length,
)
from .palette import mechanism_colour, mechanism_label, rc_params
from .static import (
    plot_amplitude_scatter,
    plot_exponent_spread,
    plot_increment_acf,
    plot_moment_spectrum,
    plot_msd,
    plot_single_trajectory_panel,
    plot_trajectories,
    plot_van_hove,
)

__all__ = [
    "palette",
    "mechanism_colour",
    "mechanism_label",
    "rc_params",
    "plot_msd",
    "plot_van_hove",
    "plot_amplitude_scatter",
    "plot_increment_acf",
    "plot_moment_spectrum",
    "plot_exponent_spread",
    "plot_single_trajectory_panel",
    "plot_trajectories",
    "plot_confusion_matrix_family",
    "plot_pairwise_error_vs_length",
    "plot_feature_importance_ranking",
    "plot_information_floor_efficiency",
    "plot_conditional_exponent_comparison",
    "AnimationStyle",
    "STYLES",
    "animate_trajectories",
    "save_animation",
    "display_animation",
]
