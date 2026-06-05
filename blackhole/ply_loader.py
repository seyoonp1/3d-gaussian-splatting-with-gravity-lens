"""Load an INRIA-format 3D Gaussian Splatting .ply into PyTorch tensors.

Handles the field layout exported by nerfstudio `ns-export gaussian-splat`:
    x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..(3*K-1), opacity, scale_0..2, rot_0..3

Applies the standard activations (INRIA stores pre-activation values):
    opacity -> sigmoid,   scale -> exp,   quaternion -> normalised
and builds per-gaussian 3x3 covariance  Sigma = R S S^T R^T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from plyfile import PlyData

C0 = 0.28209479177387814  # SH band-0 constant: rgb = 0.5 + C0 * f_dc


@dataclass
class Gaussians:
    means: torch.Tensor      # (N,3)
    scales: torch.Tensor     # (N,3) activated (world units, std-dev)
    quats: torch.Tensor      # (N,4) normalised (w,x,y,z)
    opacity: torch.Tensor    # (N,) in [0,1]
    sh: torch.Tensor         # (N, K, 3) full SH coeffs (incl. DC at index 0)
    cov: torch.Tensor        # (N,3,3) covariance
    base_rgb: torch.Tensor   # (N,3) DC-only diffuse colour in [0,1]

    def __len__(self):
        return self.means.shape[0]

    def to(self, device):
        for f in ("means", "scales", "quats", "opacity", "sh", "cov", "base_rgb"):
            setattr(self, f, getattr(self, f).to(device))
        return self


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) (w,x,y,z) -> (N,3,3). Assumes q already normalised."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.empty(N, 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_ply(path: str, device: str = "cpu", dtype=torch.float32) -> Gaussians:
    ply = PlyData.read(path)
    v = ply["vertex"]
    names = set(v.data.dtype.names)

    def col(name):
        return torch.from_numpy(np.ascontiguousarray(v[name])).to(dtype)

    means = torch.stack([col("x"), col("y"), col("z")], dim=-1)

    # SH: f_dc_0..2 (band 0) + f_rest_* (higher bands), stored channel-major per band.
    f_dc = torch.stack([col(f"f_dc_{i}") for i in range(3)], dim=-1)          # (N,3)
    rest_names = sorted([n for n in names if n.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        f_rest = torch.stack([col(n) for n in rest_names], dim=-1)           # (N, 3*K_rest)
        k_rest = f_rest.shape[-1] // 3
        # INRIA layout: [c0_coeff0..., c1_coeff..., c2_coeff...] reshaped (3, k_rest)
        f_rest = f_rest.reshape(-1, 3, k_rest).transpose(1, 2)               # (N, k_rest, 3)
    else:
        f_rest = means.new_zeros((means.shape[0], 0, 3))
    sh = torch.cat([f_dc.unsqueeze(1), f_rest], dim=1)                        # (N, K, 3)

    opacity = torch.sigmoid(col("opacity"))
    scales = torch.exp(torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1))
    quats = torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1)
    quats = torch.nn.functional.normalize(quats, dim=-1)

    R = _quat_to_rotmat(quats)
    S = torch.zeros_like(R)
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = scales[:, 0], scales[:, 1], scales[:, 2]
    M = R @ S
    cov = M @ M.transpose(1, 2)                                              # R S S^T R^T

    base_rgb = torch.clamp(0.5 + C0 * f_dc, 0.0, 1.0)

    g = Gaussians(means, scales, quats, opacity, sh, cov, base_rgb)
    return g.to(device)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "exports/treehill/splat.ply"
    g = load_ply(path, device="cpu")
    print(f"loaded {len(g):,} gaussians from {path}")
    print("SH coeffs per gaussian (incl DC):", g.sh.shape[1])
    c = g.means.mean(0)
    ext = g.means.max(0).values - g.means.min(0).values
    print(f"means bbox min {g.means.min(0).values.tolist()}")
    print(f"means bbox max {g.means.max(0).values.tolist()}")
    print(f"scene center {c.tolist()}  extent {ext.tolist()}")
    print(f"scale (world std)  min {g.scales.min().item():.4g}  "
          f"median {g.scales.median().item():.4g}  max {g.scales.max().item():.4g}")
    print(f"opacity  min {g.opacity.min().item():.3f}  "
          f"median {g.opacity.median().item():.3f}  max {g.opacity.max().item():.3f}")
    print(f"base_rgb mean {g.base_rgb.mean(0).tolist()}")
