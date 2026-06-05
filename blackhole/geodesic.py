"""Schwarzschild null-geodesic ray bending, pure PyTorch (batched).

We integrate photon paths in Cartesian coordinates around a black hole at the
origin. Using the standard orbit equation for a Schwarzschild null geodesic,

    d^2u/dphi^2 + u = (3/2) r_s u^2        (u = 1/r)

the equivalent Cartesian acceleration on the photon is

    dp/dl = v
    dv/dl = -(3/2) * r_s * h^2 * p / r^5    with  h^2 = |p x v|^2,  r = |p|

where `l` is an affine parameter and `v` is kept unit-length (null ray).  In the
weak-field limit this reproduces the Einstein deflection angle

    alpha ~= 2 r_s / b = 4 G M / (c^2 b)

(`b` = impact parameter), which `tests/test_geodesic.py` checks numerically.

`r_s` (Schwarzschild radius) is the single black-hole strength parameter; the
event-horizon shadow is `r < r_s`.
"""

from __future__ import annotations

import torch


def schwarzschild_accel(p: torch.Tensor, v: torch.Tensor, r_s: float,
                        strength: float = 1.0) -> torch.Tensor:
    """dv/dl for a Schwarzschild null geodesic.

    Args:
        p: (..., 3) photon positions relative to the black hole centre.
        v: (..., 3) photon velocities (unit length).
        r_s: Schwarzschild radius.
        strength: artistic multiplier on the deflection (1.0 = physically exact
            2 r_s/b; <1 = gentler bending, for a stylised/everyday-scene look).
    Returns:
        (..., 3) acceleration dv/dl.
    """
    r = torch.linalg.norm(p, dim=-1, keepdim=True)            # (...,1)
    h = torch.linalg.cross(p, v, dim=-1)                      # angular momentum vector
    h2 = (h * h).sum(dim=-1, keepdim=True)                    # |p x v|^2  (conserved)
    # -(3/2) r_s h^2 p / r^5 ; clamp r away from 0 to avoid blow-up inside horizon
    r5 = torch.clamp(r, min=1e-9) ** 5
    return -1.5 * strength * r_s * h2 * p / r5


def rk4_step(p: torch.Tensor, v: torch.Tensor, r_s: float, dl: float,
             strength: float = 1.0):
    """One classical RK4 step of the geodesic ODE. Returns (p_next, v_next)."""
    def deriv(p_, v_):
        return v_, schwarzschild_accel(p_, v_, r_s, strength)

    k1p, k1v = deriv(p, v)
    k2p, k2v = deriv(p + 0.5 * dl * k1p, v + 0.5 * dl * k1v)
    k3p, k3v = deriv(p + 0.5 * dl * k2p, v + 0.5 * dl * k2v)
    k4p, k4v = deriv(p + dl * k3p, v + dl * k3v)

    p_next = p + (dl / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    v_next = v + (dl / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return p_next, v_next


@torch.no_grad()
def integrate_ray(
    p0: torch.Tensor,
    v0: torch.Tensor,
    r_s: float,
    dl: float,
    n_steps: int,
    renormalize_v: bool = True,
):
    """Integrate photon paths and return the full polyline.

    Args:
        p0: (N, 3) start positions (relative to black-hole centre).
        v0: (N, 3) start directions (will be normalised to unit length).
        r_s: Schwarzschild radius.
        dl:  affine step size.
        n_steps: number of RK4 steps.
        renormalize_v: rescale v to unit length each step (null-ray hygiene).
    Returns:
        path:    (n_steps+1, N, 3) positions along each ray.
        captured:(N,) bool, True if the ray fell inside r < r_s (shadow).
    """
    p = p0.clone()
    v = torch.nn.functional.normalize(v0, dim=-1)
    N = p.shape[0]
    path = [p.clone()]
    captured = torch.zeros(N, dtype=torch.bool, device=p.device)

    for _ in range(n_steps):
        p, v = rk4_step(p, v, r_s, dl)
        if renormalize_v:
            v = torch.nn.functional.normalize(v, dim=-1)
        r = torch.linalg.norm(p, dim=-1)
        captured |= r < r_s
        path.append(p.clone())

    return torch.stack(path, dim=0), captured
