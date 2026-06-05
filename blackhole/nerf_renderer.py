"""nerf_renderer.py — render a trained nerfstudio nerfacto model with
straight rays (native, for snapshots) OR curved Schwarzschild geodesics
(for the black hole), by querying the continuous density/color field.

Why NeRF instead of the gaussian marcher: nerfacto is a continuous field, so
point-sampling along a ray gives clean output (no splat-coverage speckle). The
black hole just bends the ray; the field is queried at the bent sample points.

Coordinate frames:
  * The viewer/gsplat and our rays all use OpenCV c2w (x right, y down, z forward).
    Both render_straight and render_curved build rays with camera_rays_from_c2w in
    that frame, so an rs=0 render lines up with the gsplat preview pixel-for-pixel
    (no camera-convention guessing — render_straight feeds those rays straight into
    a nerfstudio RayBundle).
  * The nerfacto field applies SceneContraction internally in get_density, so we
    pass raw (un-contracted) world positions — same frame the model trained in
    (= the ns-export ply frame, so the black-hole position matches the gsplat view).

API (nerfstudio 1.1.5, verified):
  field.get_density(RaySamples)->(density,emb);  field.get_outputs(rs,emb)->{RGB}
  Frustums(origins,directions,starts,ends,pixel_area); starts=ends=0 => positions=origins
  model.get_outputs_for_camera_ray_bundle(RayBundle) for the native straight render.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Tensor = torch.Tensor


def camera_rays_from_c2w(c2w: Tensor, K: Tensor, W: int, H: int,
                         device="cuda") -> tuple[Tensor, Tensor]:
    """Camera rays in gsplat / OpenCV convention (x right, y down, z forward), so an
    rs=0 render lines up with the gsplat live preview. c2w [4,4], K [3,3] ->
    (origins [HW,3], directions [HW,3] unit)."""
    c2w = c2w.to(device).float()
    K = K.to(device).float()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    j, i = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32),
                          indexing="ij")
    x = (i + 0.5 - cx) / fx
    y = (j + 0.5 - cy) / fy
    z = torch.ones_like(x)
    dirs_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)        # [HW,3]
    dirs_world = torch.nn.functional.normalize(dirs_cam @ c2w[:3, :3].T, dim=-1)
    origins = c2w[:3, 3][None].expand_as(dirs_world)
    return origins.contiguous(), dirs_world.contiguous()


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
@dataclass
class NerfModel:
    model: object          # NerfactoModel (eval mode)
    device: str
    aabb: Tensor           # [2,3] scene box min/max (world)

    @property
    def center(self):
        return self.aabb.mean(0)

    @property
    def scale(self):
        return float((self.aabb[1] - self.aabb[0]).norm())


def load_nerf(config_path: str) -> NerfModel:
    from nerfstudio.utils.eval_utils import eval_setup
    _, pipeline, _, _ = eval_setup(Path(config_path), test_mode="inference")
    model = pipeline.model
    model.eval()
    dev = str(model.device)
    try:
        aabb = model.scene_box.aabb.to(dev).float()         # [2,3]
    except Exception:
        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32, device=dev)
    print(f"loaded nerfacto on {dev}; scene aabb {aabb.tolist()}")
    return NerfModel(model=model, device=dev, aabb=aabb)


# ----------------------------------------------------------------------------
# Native straight render (snapshot) — best quality (proposal + importance sampling)
# ----------------------------------------------------------------------------
@torch.no_grad()
def render_straight(nm: NerfModel, c2w, K, W: int, H: int) -> np.ndarray:
    """c2w (OpenCV) [4,4], K [3,3] -> [H,W,3] float linear RGB via nerfacto's NATIVE
    renderer (proposal importance sampling + background + appearance).

    We feed a RayBundle built from camera_rays_from_c2w (the SAME OpenCV ray formula
    gsplat uses via viewmat=inv(c2w)), so the framing is guaranteed identical to the
    gsplat preview — no camera-convention guessing. The model's collider sets
    near/far and runs proposal sampling for full quality."""
    from nerfstudio.cameras.rays import RayBundle
    dev = nm.device
    o, d = camera_rays_from_c2w(torch.tensor(np.asarray(c2w, np.float32)),
                                torch.tensor(np.asarray(K, np.float32)), W, H, device=dev)
    o = o.reshape(H, W, 3); d = d.reshape(H, W, 3)
    fx, fy = float(K[0][0]), float(K[1][1])
    pix = torch.full((H, W, 1), 1.0 / (fx * fy), device=dev)
    ci = torch.zeros((H, W, 1), dtype=torch.long, device=dev)
    rb = RayBundle(origins=o.contiguous(), directions=d.contiguous(),
                   pixel_area=pix, camera_indices=ci)
    out = nm.model.get_outputs_for_camera_ray_bundle(rb)
    return out["rgb"].clamp(0, 1).cpu().numpy()


# ----------------------------------------------------------------------------
# Field query at arbitrary points (the curved-ray building block)
# ----------------------------------------------------------------------------
@torch.no_grad()
def query_field(model, positions: Tensor, directions: Tensor):
    """positions [M,3], directions [M,3] unit -> density [M,1], rgb [M,3].
    starts=ends=0 makes Frustums.get_positions() == positions (raw world; the
    field applies SceneContraction itself)."""
    from nerfstudio.cameras.rays import RaySamples, Frustums
    from nerfstudio.field_components.field_heads import FieldHeadNames
    z = torch.zeros(positions.shape[0], 1, device=positions.device)
    fr = Frustums(origins=positions, directions=directions,
                  starts=z, ends=z, pixel_area=torch.ones_like(z))
    rs = RaySamples(frustums=fr, camera_indices=torch.zeros_like(z, dtype=torch.long))
    density, emb = model.field.get_density(rs)
    out = model.field.get_outputs(rs, emb)
    rgb = out[FieldHeadNames.RGB]
    return density, rgb


# ----------------------------------------------------------------------------
# Curved (Schwarzschild) geodesic march through the field — the black hole
# ----------------------------------------------------------------------------
@dataclass
class CurvedConfig:
    bh_center: tuple = (0.0, 0.0, 0.0)
    rs: float = 0.05            # Schwarzschild radius (world units, 0=straight)
    strength: float = 1.0       # artistic deflection multiplier
    near: float = 0.05
    far: float = 8.0
    n_steps: int = 256
    bh_near_frac: float = 5.0   # shrink dt where r < bh_near_frac*rs
    chunk: int = 1 << 18        # field/proposal-query point chunk (hash-grid is heavy)
    bg: tuple = (0.0, 0.0, 0.0) # background / captured-ray colour (space=black)


@torch.no_grad()
def render_curved(nm: NerfModel, c2w, K, W: int, H: int, cfg: CurvedConfig,
                  ss: int = 1, force_curved: bool = False) -> np.ndarray:
    """Importance-sampled curved (Schwarzschild) render. EVERY ray is geodesic-
    marched, then sampled with nerfacto's native proposal cascade along the bent
    polyline (see curved_proposal.curved_proposal_render): a coarse proposal-network
    density guides inverse-CDF importance sampling of the fine field. One renderer ->
    no hybrid-blend sky ghosting; inside the photon-sphere capture radius contributes
    nothing -> solid black shadow.

    rs<=0 -> pure native straight render (unless force_curved, for debugging the
    curved integrator itself). c2w OpenCV. Returns [H,W,3]."""
    from curved_proposal import curved_proposal_render
    dev = nm.device
    if cfg.rs <= 0.0 and not force_curved:
        return render_straight(nm, c2w, K, W, H)

    Ws, Hs = W * ss, H * ss
    Knp = np.asarray(K, np.float32).copy(); Knp[0] *= ss; Knp[1] *= ss
    o_all, d_all = camera_rays_from_c2w(torch.tensor(np.asarray(c2w, np.float32)),
                                        torch.tensor(Knp), Ws, Hs, device=dev)
    # geodesic polyline resolution >= the first proposal round (256)
    n_uniform = max(256, min(int(cfg.n_steps), 384))
    out = torch.zeros(o_all.shape[0], 3, device=dev)
    RB = 1 << 17                               # big ray batch -> fewer Python march loops
    #   (lower to 1<<14 if sharing the GPU with the viewer and you hit OOM)
    for rb in range(0, o_all.shape[0], RB):
        re = min(o_all.shape[0], rb + RB)
        out[rb:re] = curved_proposal_render(
            nm, o_all[rb:re], d_all[rb:re], cfg, query_field,
            n_uniform=n_uniform, prop_counts=(256, 96), n_field=48,
            view_blend=0.0)

    img = out.reshape(Hs, Ws, 3)
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean(dim=(1, 3))
    return img.clamp(0, 1).cpu().numpy()


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rs", type=float, default=0.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=384)
    ap.add_argument("--out", default="renders/nerf_curved.png")
    args = ap.parse_args()

    nm = load_nerf(args.config)
    # a simple camera: look at scene centre from +z (OpenCV)
    c = nm.center.cpu().numpy()
    cam = c + np.array([0, 0, nm.scale * 0.5], np.float32)
    fwd = (c - cam); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = cam
    fx = (args.W / 2) / math.tan(math.radians(50) * 0.5)
    K = np.array([[fx, 0, args.W / 2], [0, fx, args.H / 2], [0, 0, 1]], np.float32)
    cfg = CurvedConfig(bh_center=tuple(c.tolist()), rs=args.rs, strength=args.strength,
                       near=nm.scale * 0.02, far=nm.scale * 2.0)
    t0 = time.time()
    img = render_curved(nm, c2w, K, args.W, args.H, cfg)
    print(f"curved render {args.W}x{args.H} rs={args.rs} in {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio.v3 as iio
    # model RGB is already in sRGB/display space — write as-is (no extra gamma)
    iio.imwrite(args.out, (np.clip(img, 0, 1) * 255).astype(np.uint8))
    print("saved", args.out)
