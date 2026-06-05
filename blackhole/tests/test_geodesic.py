"""Validate the Schwarzschild RK4 integrator against the analytic deflection.

A photon with large impact parameter b passing a black hole of Schwarzschild
radius r_s is deflected by  alpha ~= 2 r_s / b  (= 4 G M / c^2 b).

We shoot photons in the -x direction starting far away at x = +D with various
offsets y = b, integrate until they escape, then measure the turn angle of the
velocity vector and compare to the weak-field prediction.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geodesic import integrate_ray  # noqa: E402


def deflection_for_b(b, r_s=1.0, D=4000.0, dl=2.0, n_steps=8000, dtype=torch.float64):
    """Numerically measured deflection angle (rad) for impact parameter b."""
    p0 = torch.tensor([[D, b, 0.0]], dtype=dtype)
    v0 = torch.tensor([[-1.0, 0.0, 0.0]], dtype=dtype)  # heading -x
    path, captured = integrate_ray(p0, v0, r_s=r_s, dl=dl, n_steps=n_steps,
                                   renormalize_v=True)
    # final velocity = direction of last segment
    v_final = path[-1] - path[-2]
    v_final = torch.nn.functional.normalize(v_final, dim=-1)[0]
    v_init = v0[0]
    cos_t = torch.clamp((v_final * v_init).sum(), -1.0, 1.0)
    angle = torch.arccos(cos_t)  # total turn of the velocity vector
    return float(angle), bool(captured[0])


def main():
    r_s = 1.0
    print(f"{'b/r_s':>8} {'alpha_num':>12} {'alpha_thy=2rs/b':>16} {'rel_err':>10}")
    ok = True
    for b in [200.0, 400.0, 800.0, 1600.0]:
        alpha_num, captured = deflection_for_b(b, r_s=r_s)
        alpha_thy = 2.0 * r_s / b
        rel = abs(alpha_num - alpha_thy) / alpha_thy
        print(f"{b/r_s:8.0f} {alpha_num:12.6e} {alpha_thy:16.6e} {rel:10.2%}")
        # weak-field formula is leading-order; allow a few % (higher-order in r_s/b)
        if rel > 0.03:
            ok = False

    # sanity: a photon aimed near the centre should be captured (shadow)
    _, captured = deflection_for_b(1.5 * r_s, r_s=r_s, D=200.0, dl=0.05, n_steps=20000)
    print(f"\nclose photon (b=1.5 r_s) captured (r<r_s)? {captured}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
