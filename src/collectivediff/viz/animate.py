"""Trajectory animation: three modes, one implementation.

``PHASE1_PROMPT.md`` step 4 asks for a single particle with its trail, a handful
with distinguishable trails, and a large cloud rendered as a scatter with a
subsampled trail. They differ only in how much history is drawn and how densely
frames are sampled, so they are three :class:`AnimationStyle` presets consumed by
one :func:`animate_trajectories`.

The frame-subsampling rule is not cosmetic. A thousand-step trajectory animated
at every step is a forty-second clip in which nothing legible happens; worse, for
the cloud case it is thousands of artist updates per frame for a picture that
changes imperceptibly. :attr:`AnimationStyle.max_frames` caps the frame count and
the walk is strided to fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import palette

__all__ = [
    "AnimationStyle",
    "STYLES",
    "animate_trajectories",
    "display_animation",
    "save_animation",
]

Mode = Literal["single", "few", "cloud"]


@dataclass(frozen=True)
class AnimationStyle:
    """How much of the walk to draw, and how often.

    Parameters
    ----------
    n_particles : int
        How many trajectories to render.
    trail : int or None
        Number of past positions kept behind each particle. ``None`` means the
        whole history so far -- affordable only for one or a few particles.
    trail_stride : int
        Keep every ``trail_stride``-th point of the trail. For the cloud this is
        what makes a trail affordable at all.
    max_frames : int
        Upper bound on rendered frames; the trajectory is strided to fit.
    marker_size : float
        Area of the current-position marker.
    line_width : float
        Trail width; zero draws the trail as faded points instead of a line.
    distinct_colours : bool
        Give each particle its own categorical hue. Only meaningful for the
        handful case -- ``PROJECT.md``'s palette has eight slots, and a cloud
        would be recycling them.
    alpha : float
        Opacity of the current-position markers.
    """

    n_particles: int
    trail: int | None
    trail_stride: int = 1
    max_frames: int = 240
    marker_size: float = 60.0
    line_width: float = 1.6
    distinct_colours: bool = False
    alpha: float = 0.9


#: The three modes of ``PHASE1_PROMPT.md`` step 4.
STYLES: dict[str, AnimationStyle] = {
    "single": AnimationStyle(
        n_particles=1,
        trail=None,
        max_frames=200,
        marker_size=90.0,
        line_width=1.8,
    ),
    "few": AnimationStyle(
        n_particles=8,
        trail=None,
        max_frames=180,
        marker_size=60.0,
        line_width=1.4,
        distinct_colours=True,
    ),
    "cloud": AnimationStyle(
        n_particles=3000,
        trail=40,
        trail_stride=4,
        max_frames=150,
        marker_size=6.0,
        line_width=0.0,
        alpha=0.35,
    ),
}


def _frame_indices(n_steps: int, max_frames: int) -> NDArray[np.intp]:
    """Frames to render, strided so their count stays under ``max_frames``."""
    if n_steps <= max_frames:
        return np.arange(n_steps, dtype=np.intp)
    stride = int(np.ceil(n_steps / max_frames))
    frames = np.arange(0, n_steps, stride, dtype=np.intp)
    if frames[-1] != n_steps - 1:
        frames = np.append(frames, n_steps - 1)
    return frames


def _limits(data: NDArray[np.float64], pad: float = 0.06) -> tuple[float, float]:
    """Symmetric-ish axis limits with a margin, robust to a single wild excursion.

    The 99.5th percentile rather than the maximum: a Levy flight's largest jump
    can be orders of magnitude beyond the bulk, and scaling to it would render
    every other particle as a dot at the origin.
    """
    high = float(np.percentile(np.abs(data), 99.5))
    if not np.isfinite(high) or high <= 0.0:
        high = float(np.max(np.abs(data))) or 1.0
    return -high * (1.0 + pad), high * (1.0 + pad)


def animate_trajectories(
    trajectories: NDArray[Any],
    mode: Mode | AnimationStyle = "single",
    mechanism: Any = "brownian",
    title: str | None = None,
    interval_ms: int = 40,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: float = 90.0,
) -> FuncAnimation:
    """Animate a walk in one or two dimensions.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``. In 1-D the walk is drawn
        against time on the horizontal axis; in 2-D it is drawn in the plane.
        ``PROJECT.md`` section 2 keeps two-dimensional generators for
        visualisation, which is this.
    mode : {"single", "few", "cloud"} or AnimationStyle
        Preset, or an explicit style.
    mechanism : str or GeneratorConfig
        Sets the hue when ``distinct_colours`` is off.
    title : str, optional
        Defaults to the mechanism name and the mode.
    interval_ms : int
        Delay between frames.
    figsize : tuple of float
        Figure size in inches.
    dpi : float
        Figure resolution. The default is deliberately below the project's
        static-figure setting: :func:`display_animation` embeds every frame as a
        base64 PNG, so the notebook's size on disk scales with
        ``frames * width * height * dpi^2``. At 90 dpi a two-hundred-frame clip
        costs a few megabytes; at the static 110 it costs half again as much for
        no visible gain in a looping animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Not displayed or saved; pass it to :func:`save_animation` or
        :func:`display_animation`.

    Raises
    ------
    ValueError
        If the array holds fewer particles than the style asks for.
    """
    import matplotlib.pyplot as plt

    style = STYLES[mode] if isinstance(mode, str) else mode
    trajectories = np.asarray(trajectories)
    if trajectories.ndim != 3:
        raise ValueError(f"expected (n_particles, n_steps, n_dim), got {trajectories.shape}")

    n_available, n_steps, n_dim = trajectories.shape
    n_show = min(style.n_particles, n_available)
    if n_show < 1:
        raise ValueError("no trajectories to animate")
    walk = np.asarray(trajectories[:n_show], dtype=np.float64)

    frames = _frame_indices(n_steps, style.max_frames)
    base_colour = palette.mechanism_colour(mechanism)
    colours = (
        [palette.CATEGORICAL[i % len(palette.CATEGORICAL)] for i in range(n_show)]
        if style.distinct_colours
        else [base_colour] * n_show
    )

    with plt.rc_context(palette.rc_params()):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(palette.SURFACE)

    if n_dim == 1:
        ax.set_xlim(0, n_steps - 1)
        ax.set_ylim(*_limits(walk[:, :, 0]))
        ax.set_xlabel("time $t$")
        ax.set_ylabel("$x(t)$")
        ax.axhline(0.0, color=palette.TEXT_SECONDARY, lw=0.8, zorder=1)
    else:
        # One limit for both axes, taken from the combined data. Scaling x and y
        # independently under an equal aspect ratio leaves the box squashed into
        # a band with the walk crammed into the middle of it.
        span = _limits(walk[:, :, :2])
        ax.set_xlim(*span)
        ax.set_ylim(*span)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")
        ax.plot([0.0], [0.0], marker="+", ms=8, color=palette.TEXT_SECONDARY, zorder=1)

    label = palette.mechanism_label(mechanism)
    ax.set_title(title or f"{label} ({'single' if n_show == 1 else f'{n_show}'} walkers)")

    # Trails. A line per particle for the single and handful cases, where the
    # trail is a path; one faded scatter for the cloud, where thousands of
    # LineCollections would cost far more than they show.
    trail_lines = []
    trail_cloud = None
    if style.line_width > 0.0:
        for index in range(n_show):
            (line,) = ax.plot(
                [], [], color=colours[index], lw=style.line_width, alpha=0.75, zorder=2
            )
            trail_lines.append(line)
    else:
        trail_cloud = ax.scatter(
            [], [], s=style.marker_size * 0.4, c=base_colour, alpha=0.10, lw=0, zorder=2
        )

    head = ax.scatter(
        walk[:, 0, 0],
        walk[:, 0, 1] if n_dim == 2 else np.zeros(n_show),
        s=style.marker_size,
        c=colours,
        alpha=style.alpha,
        lw=0,
        zorder=4,
    )
    clock = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=palette.TEXT_SECONDARY,
    )

    # No legend. The handful mode gives each walker its own hue so the trails
    # can be told apart, but the walkers are interchangeable samples with no
    # identity to name -- a box reading "walker 1 ... walker 8" would carry no
    # information while covering the very trails it claims to label.

    def draw(frame: int):
        start = 0 if style.trail is None else max(0, frame - style.trail)
        history = slice(start, frame + 1, style.trail_stride)

        if n_dim == 1:
            time_axis = np.arange(n_steps, dtype=np.float64)
            head.set_offsets(np.column_stack([np.full(n_show, frame), walk[:, frame, 0]]))
            if trail_lines:
                for index, line in enumerate(trail_lines):
                    line.set_data(time_axis[history], walk[index, history, 0])
            else:
                past = walk[:, history, 0]
                times = np.broadcast_to(time_axis[history], past.shape)
                trail_cloud.set_offsets(
                    np.column_stack([times.ravel(), past.ravel()])
                )
        else:
            head.set_offsets(walk[:, frame, :])
            if trail_lines:
                for index, line in enumerate(trail_lines):
                    line.set_data(walk[index, history, 0], walk[index, history, 1])
            else:
                past = walk[:, history, :]
                trail_cloud.set_offsets(past.reshape(-1, 2))

        clock.set_text(f"$t$ = {frame}")
        return [head, clock, *trail_lines] + ([trail_cloud] if trail_cloud else [])

    return FuncAnimation(
        fig, draw, frames=frames, interval=interval_ms, blit=False, repeat=True
    )


def save_animation(
    animation: FuncAnimation,
    path: str | Path,
    fps: int = 25,
    dpi: int = 110,
) -> Path:
    """Write an animation to mp4, falling back to an animated gif.

    mp4 needs ``ffmpeg`` on the path. When it is missing -- a fresh machine, a
    CI container -- writing nothing at all would be the worst outcome, so the
    suffix is swapped for ``.gif`` and Pillow renders it instead. The returned
    path is the one actually written, which may not be the one requested.

    Parameters
    ----------
    animation : matplotlib.animation.FuncAnimation
        Animation to write.
    path : path-like
        Target file. A ``.mp4`` suffix requests video.
    fps : int
        Frames per second.
    dpi : int
        Output resolution.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".mp4" and FFMpegWriter.isAvailable():
        animation.save(path, writer=FFMpegWriter(fps=fps), dpi=dpi)
        return path

    gif_path = path.with_suffix(".gif")
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    return gif_path


def display_animation(animation: FuncAnimation) -> Any:
    """Wrap an animation for inline display in a notebook.

    Returns an ``IPython.display.HTML`` holding a JavaScript-driven player, so a
    notebook shows the animation without a video codec and without leaving a
    file behind.

    Parameters
    ----------
    animation : matplotlib.animation.FuncAnimation

    Returns
    -------
    IPython.display.HTML
    """
    from IPython.display import HTML

    return HTML(animation.to_jshtml(default_mode="loop"))
