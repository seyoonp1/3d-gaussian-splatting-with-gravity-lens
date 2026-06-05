"""curved_proposal.py — native-equivalent proposal sampling along a curved
(geodesic) path, for nerf_renderer.render_curved.

This reproduces nerfacto's exact sampling pipeline, but along the BENT ray, so an
rs=0 curved render matches the native straight renderer:

  * One dense geodesic polyline carries the ray geometry (positions vs arc-length).
    Bending only happens near the black hole, so the polyline is marched finely
    there over [near, cfg.far] and then extended STRAIGHT to the far plane (no
    bending that far out — one extra vertex at far is exact for the straight tail).
  * All SAMPLING happens in nerfstudio's spacing coordinate s in [0,1] with the
    piecewise linear-near / disparity-far mapping (UniformLinDispPiecewiseSampler),
    over the model collider's near/far (0.05 .. 1000). Samples therefore bunch near
    the camera and stretch into the contracted background exactly like native.
  * Cascade (nerfacto-exact): uniform-in-s 256 -> density_fns[0] -> PDF resample in
    s-space -> 96 -> density_fns[1] -> PDF resample -> 48 -> main field. Sampling is
    deterministic (eval mode: nerfstudio only jitters while training).
  * Composite front-to-back with finite euclidean bin widths and a "last_sample"
    background (native RGBRenderer); captured (shadow) rays composite to black.

Coordinate frames & field semantics are unchanged from nerf_renderer.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from geodesic import rk4_step

Tensor = torch.Tensor
EPS = 1e-9


# ---------------------------------------------------------------------------
# polyline helpers (positions vs arc-length; positions via linear interpolation)
# ---------------------------------------------------------------------------
def interp_polyline(positions: Tensor, arclen: Tensor, s: Tensor):
    """positions [B,V,3], arclen [B,V] (increasing), s [B,M] query arc-lengths
    -> pos [B,M,3], tan [B,M,3] (unit). Linear interp; clamps outside the polyline."""
    V = arclen.shape[1]
    hi = torch.searchsorted(arclen, s, right=True).clamp(1, V - 1)
    lo = hi - 1
    a_lo = torch.gather(arclen, 1, lo)
    a_hi = torch.gather(arclen, 1, hi)
    fr = ((s - a_lo) / (a_hi - a_lo + EPS)).clamp(0, 1)[..., None]
    p_lo = torch.gather(positions, 1, lo[..., None].expand(-1, -1, 3))
    p_hi = torch.gather(positions, 1, hi[..., None].expand(-1, -1, 3))
    pos = p_lo + fr * (p_hi - p_lo)
    tan = F.normalize(p_hi - p_lo, dim=-1)        # local path direction
    return pos, tan


# ---------------------------------------------------------------------------
# nerfstudio spacing (UniformLinDispPiecewise): s in [0,1] <-> euclidean distance
# ---------------------------------------------------------------------------
def _spacing_fn(t: Tensor) -> Tensor:
    return torch.where(t < 1, t / 2, 1 - 1 / (2 * t))


def _spacing_fn_inv(y: Tensor) -> Tensor:
    return torch.where(y < 0.5, 2 * y, 1 / (2 * (1 - y)))


def sample_pdf(edges: Tensor, weights: Tensor, num_samples: int,
               padding: float = 0.01) -> Tensor:
    """Native PDFSampler (eval mode): resample spacing `edges` [B,Nin+1] by `weights`
    [B,Nin] into num_samples+1 new edges. Deterministic stratified centres (no jitter,
    matching inference), histogram_padding=0.01, include_original=False."""
    dev = edges.device
    B = edges.shape[0]
    w = weights + padding
    wsum = w.sum(-1, keepdim=True)
    pad = torch.relu(1e-5 - wsum)                          # guard all-zero weights
    w = w + pad / w.shape[-1]
    wsum = wsum + pad
    pdf = w / wsum
    cdf = torch.clamp(torch.cumsum(pdf, -1), max=1.0)
    cdf = torch.cat([torch.zeros(B, 1, device=dev), cdf], -1)        # [B,Nin+1]
    nb = num_samples + 1
    u = torch.linspace(0.0, 1.0 - 1.0 / nb, nb, device=dev) + 1.0 / (2 * nb)
    u = u[None].expand(B, nb).contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(0, edges.shape[1] - 1)
    above = inds.clamp(0, edges.shape[1] - 1)
    cdf_b = torch.gather(cdf, 1, below); cdf_a = torch.gather(cdf, 1, above)
    e_b = torch.gather(edges, 1, below); e_a = torch.gather(edges, 1, above)
    denom = (cdf_a - cdf_b)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_b) / denom
    return e_b + t * (e_a - e_b)


# ---------------------------------------------------------------------------
# dense geodesic polyline (fine near the BH, straight tail to the far plane)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_geodesic_polyline(o, d, bh, cfg, n_uniform: int, near: float,
                            far_sample: float, capture_frac: float = 1.5):
    """o,d [B,3] (world, OpenCV). March a dense geodesic polyline over [near, cfg.far]
    (finer dt near the BH), then append one far vertex at arc-length far_sample by
    straight extrapolation. Returns positions [B,V,3] (world), arclen [B,V],
    cap_s [B] (arc-length where captured; 1e9 if never).

    Capture radius = capture_frac * r_s (default 1.5 = the PHOTON SPHERE). Rays that
    reach the photon sphere are doomed (b < b_crit), so we capture them there: this
    (a) gives the physically-correct apparent shadow (b_crit = 2.6 r_s), and (b)
    stops the geodesic from overshooting in the huge-acceleration small-r region."""
    dev = o.device
    B = o.shape[0]
    rs_cap = cfg.rs * capture_frac
    vel = F.normalize(d, dim=-1)
    pos = (o - bh) + vel * near                          # BH-centred, start at near plane
    positions = [pos + bh]
    arclen = [torch.full((B,), near, device=dev)]
    s_acc = torch.full((B,), near, device=dev)
    cap_s = torch.full((B,), 1e9, device=dev)
    frozen = torch.zeros(B, dtype=torch.bool, device=dev)
    dt0 = (cfg.far - near) / n_uniform                   # dense extent = cfg.far (scene field)
    for _ in range(n_uniform):
        r = pos.norm(dim=-1, keepdim=True)
        # finer dt as the ray nears the BH so rk4 doesn't overshoot the capture sphere
        dt = torch.where(r < cfg.bh_near_frac * cfg.rs,
                         torch.clamp(0.15 * r, max=dt0 * 0.25), torch.full_like(r, dt0))
        new_pos, new_vel = rk4_step(pos, vel, cfg.rs, dt, cfg.strength)
        new_vel = F.normalize(new_vel, dim=-1)
        new_pos = torch.nan_to_num(new_pos, nan=0.0, posinf=0.0, neginf=0.0)
        new_pos = torch.where(frozen[:, None], pos, new_pos)
        new_vel = torch.where(frozen[:, None], vel, new_vel)
        s_acc = s_acc + (new_pos - pos).norm(dim=-1)
        rn = new_pos.norm(dim=-1)
        newly = (rn < rs_cap) & (~frozen)
        cap_s = torch.where(newly, s_acc, cap_s)
        frozen = frozen | (rn < rs_cap)
        positions.append(new_pos + bh)
        arclen.append(s_acc.clone())
        pos, vel = new_pos, new_vel
    # one straight far vertex out to far_sample (disparity-spaced tail; no bending here)
    p_far = torch.nan_to_num((pos + bh) + vel * (far_sample - s_acc[:, None]))
    positions.append(p_far)
    arclen.append(torch.full((B,), far_sample, device=dev))
    return torch.stack(positions, 1), torch.stack(arclen, 1), cap_s


# ---------------------------------------------------------------------------
# native-equivalent cascade (all sampling in spacing s-space)
# ---------------------------------------------------------------------------
@torch.no_grad()
def curved_proposal_render(nm, o, d, cfg, query_field,
                           n_uniform=256, prop_counts=(256, 96), n_field=48,
                           view_blend=0.0, capture_frac=1.5):
    """Render a batch of curved rays with nerfacto's native proposal cascade.

    nm           : NerfModel (nm.model.density_fns = [prop0, prop1]; nm.model.collider)
    o,d          : [B,3] world OpenCV rays (already supersampled if needed)
    query_field  : nerf_renderer.query_field (positions,dirs)->(density,rgb)
    n_uniform    : dense geodesic polyline steps (>= prop_counts[0] for resolution)
    prop_counts  : initial + per-proposal sample counts (native: 256 then 96)
    n_field      : final samples into the main field (native: 48)
    view_blend   : 0 = bent tangent for colour; ->1 = straight cam->point dir (suppress
                   SH/dir extrapolation, i.e. photon-ring colour speckle, when lensed).

    Returns rgb [B,3].
    """
    dev = o.device
    B = o.shape[0]
    bh = torch.tensor(cfg.bh_center, device=dev, dtype=torch.float32)
    cam_o = o.clone()
    density_fns = nm.model.density_fns          # [prop0, prop1]
    rs_cap = cfg.rs * capture_frac              # photon-sphere capture radius

    # same near/far the native renderer samples over (NearFarCollider)
    coll = getattr(nm.model, "collider", None)
    near = float(getattr(coll, "near_plane", 0.05))
    far_sample = float(getattr(coll, "far_plane", 1000.0))

    positions, arclen, cap_s = build_geodesic_polyline(
        o, d, bh, cfg, n_uniform, near, far_sample, capture_frac=capture_frac)

    # spacing coordinate s in [0,1] -> euclidean distance t (piecewise lin/disp)
    s_near = _spacing_fn(torch.tensor(near, device=dev))
    s_far = _spacing_fn(torch.tensor(far_sample, device=dev))

    def s_to_t(x):
        return _spacing_fn_inv(x * s_far + (1 - x) * s_near)

    def query_density(dfn, t_mid):
        """proposal density at the euclidean midpoints t_mid [B,M] (chunked)."""
        pos_mid, _ = interp_polyline(positions, arclen, t_mid)
        flat = pos_mid.reshape(-1, 3)
        sig = torch.empty(flat.shape[0], 1, device=dev)
        for s0 in range(0, flat.shape[0], cfg.chunk):
            e0 = min(flat.shape[0], s0 + cfg.chunk)
            sig[s0:e0] = dfn(flat[s0:e0])
        sig = sig.reshape(B, -1)
        rmid = (pos_mid - bh).norm(dim=-1)           # kill inside horizon / past capture
        return torch.where((rmid < rs_cap) | (t_mid > cap_s[:, None]),
                           torch.zeros_like(sig), sig)

    # ---- cascade: uniform-in-s -> prop0 -> pdf -> prop1 -> pdf -> n_field ----
    N0 = prop_counts[0]
    x_edges = torch.linspace(0.0, 1.0, N0 + 1, device=dev)[None].expand(B, N0 + 1).contiguous()
    draws = list(prop_counts[1:]) + [n_field]   # e.g. (256,96)+48 -> [96, 48]
    for i, dfn in enumerate(density_fns):
        t_edges = s_to_t(x_edges)                                # euclidean bin edges
        t_mid = 0.5 * (t_edges[:, :-1] + t_edges[:, 1:])
        delta = (t_edges[:, 1:] - t_edges[:, :-1]).clamp(min=0)
        sig = query_density(dfn, t_mid)
        # native get_weights = alpha * transmittance (NOT raw alpha): the transmittance
        # is essential — without it, disparity's huge far-bin deltas make alpha~1 in the
        # background and pull samples out to the far sky instead of onto the first surface.
        alpha = 1.0 - torch.exp(-torch.relu(sig) * delta)
        Tw = torch.cumprod(torch.cat([torch.ones(B, 1, device=dev), 1 - alpha + 1e-10], 1), 1)[:, :-1]
        w = alpha * Tw
        x_edges = sample_pdf(x_edges, w, draws[i])               # resample in s-space

    # ---- main field on the final n_field samples ----
    t_edges = s_to_t(x_edges)                                    # [B, n_field+1]
    t_mid = 0.5 * (t_edges[:, :-1] + t_edges[:, 1:])
    delta = (t_edges[:, 1:] - t_edges[:, :-1]).clamp(min=0)
    M = t_mid.shape[1]
    pos_f, tan_f = interp_polyline(positions, arclen, t_mid)
    if view_blend > 0.0:
        straight = F.normalize(pos_f - cam_o[:, None, :], dim=-1)
        dirq = F.normalize((1 - view_blend) * tan_f + view_blend * straight, dim=-1)
    else:
        dirq = tan_f

    pf = pos_f.reshape(-1, 3); df = dirq.reshape(-1, 3)
    dens = torch.empty(pf.shape[0], 1, device=dev)
    col = torch.empty(pf.shape[0], 3, device=dev)
    for s0 in range(0, pf.shape[0], cfg.chunk):
        e0 = min(pf.shape[0], s0 + cfg.chunk)
        dc, cc = query_field(nm.model, pf[s0:e0], df[s0:e0])
        dens[s0:e0] = dc; col[s0:e0] = cc
    dens = dens.reshape(B, M); col = col.reshape(B, M, 3)

    rr = (pos_f - bh).norm(dim=-1)
    dens = torch.where((rr < rs_cap) | (t_mid > cap_s[:, None]), torch.zeros_like(dens), dens)

    alpha = 1.0 - torch.exp(-torch.relu(dens) * delta)
    T = torch.cumprod(torch.cat([torch.ones(B, 1, device=dev), 1 - alpha + 1e-10], 1), 1)[:, :-1]
    weights = alpha * T
    acc = weights.sum(1, keepdim=True)
    rgb = (weights[..., None] * col).sum(1)
    # native RGBRenderer background_color="last_sample"; captured (shadow) rays -> black
    captured = (cap_s < far_sample * 0.999)[:, None].float()
    bg = (1.0 - captured) * col[:, -1, :]
    return rgb + (1.0 - acc) * bg
