"""Optim utils.

This module contains the building blocks for end-to-end gradient-based
optimisation of antenna-array layouts using the differentiable
Kaiser-Bessel gridding forward model from :mod:`argosim.imaging_utils`.

It is intentionally modular so that the user can mix and match:

* a forward model parameterised by an :class:`ObservationConfig`;
* a target / objective loss (currently: image-matching MSE; future:
  FWHM, sidelobe level, UV-coverage metrics);
* an arbitrary list of weighted soft constraints
  (:class:`Constraint`) — minimum antenna spacing, maximum array radius,
  centroid pin, etc.;
* an optimiser built via :func:`make_optimizer` (Adam, SGD, AdamW, ...)
  with an optional learning-rate schedule via :func:`exp_decay_schedule`.

The full optimisation loop is exposed as :func:`optimise_array`, which
returns the final positions, the loss history, and per-step antenna
snapshots for animation.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from argosim import antenna_utils as aat
from argosim import imaging_utils as aiu


########################################
#  Observation configuration           #
########################################


@dataclass(frozen=True)
class ObservationConfig:
    """Observation configuration.

    Static parameters of the imaging / observation model used by the
    differentiable forward map. Kept frozen so it is hashable and safe
    to pass through :func:`jax.jit` as a static argument.

    Parameters
    ----------
    npx : int
        Number of pixels per side of the (square) image grid.
    fov : float
        Field of view in degrees (square FOV).
    lat : float
        Array latitude in radians.
    dec : float
        Source declination in radians.
    freq : float
        Central observation frequency in Hz.
    bw : float
        Total bandwidth in Hz.
    n_freqs : int
        Number of frequency channels sampled across ``bw``.
    n_times : int
        Number of time samples across the integration.
    track_h : float
        Total integration time in hours.
    t_0 : float
        Initial hour-angle in hours.
    kernel_W : int
        KB gridding kernel support width in pixels (odd integer).
    """

    npx: int = 256
    fov: float = 0.4
    lat: float = float(35.0 * np.pi / 180)
    dec: float = float(32.0 * np.pi / 180)
    freq: float = 2.0e9
    bw: float = 1.0e9
    n_freqs: int = 5
    n_times: int = 5
    track_h: float = 0.5
    t_0: float = 1.0
    kernel_W: int = 7

    @property
    def kernel_beta(self) -> float:
        """KB shape parameter (``2.34 * W``, standard choice)."""
        return 2.34 * self.kernel_W

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Image grid shape ``(npx, npx)``."""
        return (self.npx, self.npx)


########################################
#  Differentiable forward model        #
########################################


def antenna_to_beam(antenna, config: ObservationConfig):
    """Antenna positions to normalised dirty beam.

    Differentiable forward model mapping antenna ENU positions to the
    peak-normalised dirty beam (PSF) via baseline computation, UV-track
    generation, Kaiser-Bessel convolutional gridding, and inverse FFT.

    Parameters
    ----------
    antenna : jnp.ndarray
        Antenna ENU positions, shape ``(n_ant, 3)`` in metres.
    config : ObservationConfig
        Static imaging / observation parameters.

    Returns
    -------
    beam : jnp.ndarray
        Peak-normalised dirty beam, shape ``config.grid_shape``.
    """
    gs = config.grid_shape
    W = config.kernel_W
    beta = config.kernel_beta

    b_enu = aat.get_baselines(antenna)
    track, _ = aat.uv_track_multiband(
        b_ENU=b_enu,
        lat=config.lat,
        dec=config.dec,
        track_time=config.track_h,
        t_0=config.t_0,
        n_times=config.n_times,
        f=config.freq,
        df=config.bw,
        n_freqs=config.n_freqs,
        multi_band=False,
    )
    uv_px = aiu.scale_uv_samples_continuous(track, gs, (config.fov, config.fov))
    psf_grid = aiu.grid_visibilities_conv(
        jnp.ones(uv_px.shape[0], dtype=jnp.complex128), uv_px, gs, W, beta,
    )
    corr = aiu.kb_correction(gs, W, beta)
    beam = aiu.uv2sky(psf_grid) * corr

    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(beam)) + 1e-12)
    return jnp.abs(beam) / peak


########################################
#  Target / objective loss functions   #
########################################


def beam_mse_loss(beam, target):
    """Beam MSE loss.

    Normalised mean squared error between a (peak-normalised) dirty
    beam and a target image.

    Parameters
    ----------
    beam : jnp.ndarray
        Peak-normalised dirty beam.
    target : jnp.ndarray
        Target image (e.g. a tight Gaussian).

    Returns
    -------
    loss : jnp.ndarray
        Scalar normalised MSE.
    """
    return jnp.mean((target - beam) ** 2) / (jnp.max(target) ** 2)


########################################
#  Physical soft constraints           #
########################################


def spacing_penalty(antenna, d_min):
    """Spacing penalty.

    Smooth ReLU barrier on antenna pairs closer than ``d_min`` in the
    E-N plane. Each violating pair contributes ``(1 - dist/d_min)^2``.

    Parameters
    ----------
    antenna : jnp.ndarray
        Antenna ENU positions, shape ``(n_ant, 3)``.
    d_min : float
        Minimum allowed pairwise distance in metres.

    Returns
    -------
    penalty : jnp.ndarray
        Scalar non-negative penalty.
    """
    diffs = antenna[:, None, :2] - antenna[None, :, :2]
    # Diagonal masked with a large value so it never triggers the barrier.
    dist = jnp.sqrt(
        jnp.sum(diffs ** 2, axis=-1) + jnp.eye(antenna.shape[0]) * 1e12
    )
    # Pairs counted twice (i<j and j<i) → divide by 2.
    return 0.5 * jnp.sum(jax.nn.relu(1.0 - dist / d_min) ** 2)


def radius_penalty(antenna, r_max):
    """Radius penalty.

    Smooth ReLU barrier on each antenna's distance from the array
    centroid above ``r_max``. Models a circular site footprint.

    Parameters
    ----------
    antenna : jnp.ndarray
        Antenna ENU positions, shape ``(n_ant, 3)``.
    r_max : float
        Maximum allowed radius from the centroid in metres.

    Returns
    -------
    penalty : jnp.ndarray
        Scalar non-negative penalty.
    """
    centroid = jnp.mean(antenna[:, :2], axis=0)
    r = jnp.sqrt(jnp.sum((antenna[:, :2] - centroid) ** 2, axis=-1) + 1e-12)
    return jnp.sum(jax.nn.relu(r / r_max - 1.0) ** 2)


def centroid_penalty(antenna):
    """Centroid penalty.

    Squared offset of the array centroid from the origin (E-N plane).
    Removes the loss's translational degeneracy by pinning the centroid
    so the optimiser does not drift as a rigid body.

    Parameters
    ----------
    antenna : jnp.ndarray
        Antenna ENU positions, shape ``(n_ant, 3)``.

    Returns
    -------
    penalty : jnp.ndarray
        Scalar non-negative penalty in m^2.
    """
    return jnp.sum(jnp.mean(antenna[:, :2], axis=0) ** 2)


########################################
#  Composing the total loss            #
########################################


@dataclass
class Constraint:
    """Constraint.

    A weighted soft constraint applied to the antenna array.

    Parameters
    ----------
    name : str
        Human-readable name (used in reports / logs).
    fn : Callable
        ``(antenna,) -> scalar`` penalty function.
    weight : float
        Multiplicative weight in the total loss.
    """

    name: str
    fn: Callable
    weight: float


def make_loss_fn(forward_fn, target_loss_fn, constraints: Sequence[Constraint]):
    """Make loss fn.

    Build a scalar loss combining a target / objective term and a set
    of weighted soft constraints.

    Parameters
    ----------
    forward_fn : Callable
        ``(antenna,) -> observable`` — differentiable. Typically a
        partially applied :func:`antenna_to_beam`.
    target_loss_fn : Callable
        ``(observable,) -> scalar`` target term.
    constraints : sequence of Constraint
        Weighted soft constraints on antenna positions.

    Returns
    -------
    loss_fn : Callable
        ``(antenna,) -> scalar`` total loss.
    """
    def loss_fn(antenna):
        observable = forward_fn(antenna)
        L = target_loss_fn(observable)
        for c in constraints:
            L = L + c.weight * c.fn(antenna)
        return L

    return loss_fn


########################################
#  Optimiser builders                  #
########################################


def exp_decay_schedule(
    init_lr=25.0, end_lr=2.0, transition_steps=200, decay_rate=0.5
):
    """Exp decay schedule.

    Exponential learning-rate decay from ``init_lr`` toward ``end_lr``.

    Parameters
    ----------
    init_lr : float
        Initial learning rate.
    end_lr : float
        Asymptotic learning rate (lower bound).
    transition_steps : int
        Number of steps over which to apply one decay factor.
    decay_rate : float
        Multiplicative decay per ``transition_steps``.

    Returns
    -------
    schedule : optax.Schedule
        Step-indexed learning-rate schedule.
    """
    return optax.exponential_decay(
        init_value=init_lr,
        end_value=end_lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
    )


def make_optimizer(name="adam", lr=1.0, **kwargs):
    """Make optimizer.

    Build an :mod:`optax` optimiser by name.

    Parameters
    ----------
    name : str
        Optimiser name: ``"adam"``, ``"sgd"``, or ``"adamw"``.
    lr : float or optax.Schedule
        Constant learning rate or an ``optax`` schedule.
    **kwargs
        Forwarded to the underlying ``optax`` constructor.

    Returns
    -------
    optimizer : optax.GradientTransformation
        The configured optimiser.
    """
    name = name.lower()
    if name == "adam":
        return optax.adam(learning_rate=lr, **kwargs)
    if name == "sgd":
        return optax.sgd(learning_rate=lr, **kwargs)
    if name == "adamw":
        return optax.adamw(learning_rate=lr, **kwargs)
    raise ValueError(f"Unknown optimiser: {name!r}")


########################################
#  Training loop                       #
########################################


def optimise_array(
    antenna_init,
    loss_fn,
    optimizer,
    n_steps,
    snapshot_every: Optional[int] = None,
    snapshot_steps: Optional[Sequence[int]] = None,
    freeze_u: bool = True,
    verbose_every: Optional[int] = 50,
):
    """Optimise array.

    Run gradient-based optimisation of the antenna layout.

    Parameters
    ----------
    antenna_init : jnp.ndarray
        Initial antenna ENU positions, shape ``(n_ant, 3)``.
    loss_fn : Callable
        ``(antenna,) -> scalar`` total loss, e.g. from
        :func:`make_loss_fn`.
    optimizer : optax.GradientTransformation
        The optimiser to step with, e.g. from :func:`make_optimizer`.
    n_steps : int
        Number of optimisation steps to run.
    snapshot_every : int, optional
        If given, store antenna positions and loss every this many
        steps (uniform spacing).
    snapshot_steps : sequence of int, optional
        Explicit step indices at which to store a snapshot. Useful for
        non-uniform spacing (e.g. log-spaced: dense early, sparse late).
        Overrides ``snapshot_every`` when provided. Step 0 (initial)
        and step ``n_steps`` (final) are always included.
    freeze_u : bool
        If True, zero out the U (vertical) gradient component so the
        array stays flat.
    verbose_every : int, optional
        Print a status line every this many steps. Set to ``None`` or
        ``0`` to silence.

    Returns
    -------
    result : dict
        Keys: ``antenna`` (final positions, jnp.ndarray),
        ``loss_history`` (np.ndarray, length ``n_steps``),
        ``snapshots_antenna`` (np.ndarray, shape ``(n_frames, n_ant, 3)``),
        ``snapshots_loss`` (np.ndarray, length ``n_frames``),
        ``snapshots_step`` (np.ndarray of int, length ``n_frames``,
        the step index of each stored snapshot),
        ``snapshot_every`` (int or None).
    """
    # Decide the snapshot policy
    if snapshot_steps is not None:
        snap_set = {int(s) for s in snapshot_steps if 0 < int(s) <= n_steps}
        def is_snapshot_step(k):
            return k in snap_set
    elif snapshot_every is not None:
        def is_snapshot_step(k):
            return k % snapshot_every == 0
    else:
        def is_snapshot_step(k):
            return False

    @jax.jit
    def step_fn(antenna, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn)(antenna)
        if freeze_u:
            grads = grads.at[:, 2].set(0.0)
        updates, opt_state = optimizer.update(grads, opt_state)
        antenna = optax.apply_updates(antenna, updates)
        return antenna, opt_state, loss_val

    antenna = antenna_init
    opt_state = optimizer.init(antenna)
    loss_history = []
    snap_ant = [np.array(antenna)]
    snap_loss = [float(loss_fn(antenna))]
    snap_step = [0]

    for s in range(n_steps):
        antenna, opt_state, loss_val = step_fn(antenna, opt_state)
        loss_history.append(float(loss_val))
        k = s + 1
        if is_snapshot_step(k):
            snap_ant.append(np.array(antenna))
            snap_loss.append(float(loss_val))
            snap_step.append(k)
        if verbose_every and s % verbose_every == 0:
            print(f"step {s:4d}   loss = {float(loss_val):.5f}")

    # Always include the final state.
    if snap_step[-1] != n_steps:
        snap_ant.append(np.array(antenna))
        snap_loss.append(float(loss_history[-1]))
        snap_step.append(n_steps)

    return {
        "antenna": antenna,
        "loss_history": np.array(loss_history),
        "snapshots_antenna": np.stack(snap_ant),
        "snapshots_loss": np.array(snap_loss),
        "snapshots_step": np.array(snap_step),
        "snapshot_every": snapshot_every,
    }
