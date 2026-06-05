"""Interactive viser viewer (port 7007).

Architecture (final):
  * LIVE PREVIEW  = gsplat rasterization of the .ply (fast straight-ray, for orbiting).
  * 📸 SNAPSHOT   = nerfacto NATIVE render (clean continuous field) when r_s=0;
                    nerfacto CURVED geodesic render (black hole) when r_s>0.
  * 🎬 VIDEO      = nerfacto curved geodesic orbit.
  * 📊 COMPARE    = gsplat vs nerfacto-straight (coordinate-alignment check).

The .ply (ns-export splatfacto) and the nerfacto model share the SAME nerfstudio
normalized frame (same ColmapDataParser), so the draggable black-hole gizmo is in
the coordinate frame both renderers use.

Run:
    CUDA_HOME=/home/ubuntu/miniconda3/envs/3DGS python blackhole/viewer.py \
        --ply exports/treehill/splat.ply \
        --config outputs/treehill/nerfacto/<date>/config.yml --share
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import torch
import viser
import nerfview
import gsplat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ply_loader import load_ply          # noqa: E402
import nerf_renderer as nf               # noqa: E402


def estimate_up(means: torch.Tensor) -> np.ndarray:
    c = means.median(0).values
    d = (means - c).abs()
    thr = torch.quantile(d[:: max(1, means.shape[0] // 200000)], 0.85, dim=0)
    keep = (d < thr).all(dim=1)
    pts = means[keep] if int(keep.sum()) > 1000 else means
    pts = pts - pts.mean(0)
    cov = (pts.T @ pts) / pts.shape[0]
    _, evecs = torch.linalg.eigh(cov)
    up = evecs[:, 0]
    return (up / up.norm()).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="exports/treehill/splat.ply")
    ap.add_argument("--config", required=True, help="nerfacto config.yml")
    ap.add_argument("--port", type=int, default=7007)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--outdir", default="renders")
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(args.outdir, exist_ok=True)

    # ---- gsplat preview assets -------------------------------------------------
    g = load_ply(args.ply, device=device)
    sh = g.sh.contiguous().float()
    sh_degree = int(round(math.sqrt(sh.shape[1]) - 1))
    center = g.means.median(0).values            # robust (floaters)
    core = float((g.means.quantile(0.9, 0) - g.means.quantile(0.1, 0)).norm())
    print(f"loaded {len(g):,} gaussians; sh_degree {sh_degree}; core {core:.3f}")

    # ---- nerfacto model (snapshot + black hole) --------------------------------
    print("loading nerfacto…", flush=True)
    nm = nf.load_nerf(args.config)
    # curved-march near/far in the NERF's own (normalized) scale — NOT the ply core,
    # which is inflated by background floaters (=> dt too coarse, surfaces skipped).
    bh_near = nm.scale * 0.006        # ~0.02 in nerfacto normalized space
    bh_far = nm.scale * 2.3           # ~8 ; reaches the contracted background

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    if args.share:
        print(f"SHARE URL: {server.request_share_url()}", flush=True)

    # ---- world up-axis fix -----------------------------------------------------
    up_pca = estimate_up(g.means)
    print(f"estimated up (pca): {up_pca.round(3).tolist()}", flush=True)

    def apply_up(choice):
        if choice == "pca":
            server.scene.set_up_direction(tuple(float(x) for x in up_pca))
        else:
            server.scene.set_up_direction(choice)

    with server.gui.add_folder("World"):
        up_dropdown = server.gui.add_dropdown(
            "up axis", ("pca", "+z", "-z", "+y", "-y", "+x", "-x"), initial_value="pca")

    @up_dropdown.on_update
    def _(_):
        apply_up(up_dropdown.value)
    apply_up("pca")

    # ---- black-hole controls ---------------------------------------------------
    B_CRIT = 1.5 * math.sqrt(3.0)
    state = {"r_s": core * 0.05, "strength": 1.0, "busy": False,
             "last_render": None, "last_cam": None, "rs_keyframes": []}
    gizmo = server.scene.add_transform_controls(
        "/blackhole", scale=core * 0.15, line_width=2.0,
        disable_rotations=True, disable_sliders=True)
    gizmo.position = center.cpu().numpy()

    with server.gui.add_folder("Black hole"):
        rs_slider = server.gui.add_slider("r_s (size, 0=off)", min=0.0, max=core * 0.4,
                                          step=core * 0.005, initial_value=state["r_s"])
        str_slider = server.gui.add_slider("lens strength", min=0.0, max=3.0,
                                           step=0.05, initial_value=state["strength"])

    with server.gui.add_folder("NeRF render"):
        res_dd = server.gui.add_dropdown(
            "resolution (width)",
            ("384", "512", "720", "960", "1280", "1600", "1920", "2560", "3200"),
            initial_value="960",
            hint="NeRF is continuous — render any resolution. High res: native fast, "
                 "black-hole (curved march) is slow.")
        steps_slider = server.gui.add_slider("curved steps", min=128, max=640, step=64,
                                             initial_value=512)
        ss_dd = server.gui.add_dropdown("supersample (curved)", ("1", "2"), initial_value="1")
        nframes = server.gui.add_slider("video frames", min=12, max=240, step=12,
                                        initial_value=60)
        orbit_deg = server.gui.add_slider("video orbit (deg)", min=30, max=360, step=30,
                                          initial_value=360)
        snap_btn = server.gui.add_button("📸 Render snapshot (NeRF)")
        cmp_btn = server.gui.add_button("📊 Compare gsplat ↔ NeRF (this view)")
        ncmp_btn = server.gui.add_button("🔬 Compare native ↔ curved(rs=0)")
        vid_btn = server.gui.add_button("🎬 Render black-hole orbit video")
        status = server.gui.add_text("status", initial_value="ready", disabled=True)
        progress = server.gui.add_progress_bar(0.0, animated=False)
        show_result = server.gui.add_checkbox("🖼 show last render (freeze)", initial_value=False)

    # Use the Rendering panel's "Add Keyframe" to set keyframes (camera + r_s), then
    # render here directly (reliable — bypasses viser's client-side get_render).
    with server.gui.add_folder("Black hole video (from Rendering keyframes)"):
        rskf_info = server.gui.add_text("r_s per keyframe", initial_value="(none)",
                                        disabled=True)
        rskf_hint = server.gui.add_text(
            "how-to", initial_value="set r_s → Rendering→Add Keyframe (x2+) → render",
            disabled=True)
        rskf_hint2 = server.gui.add_text(
            "frames", initial_value="uses Rendering panel's FPS x Duration", disabled=True)
        renderkf_btn = server.gui.add_button("🎬 Render black-hole video")

    @show_result.on_update
    def _(_):
        rerender()

    # ---- live preview: gsplat + BH marker --------------------------------------
    @torch.no_grad()
    def render_fn(camera_state, img_wh):
        W, H = img_wh
        state["last_cam"] = camera_state          # the camera actually being viewed
        # frozen result mode: show the last NeRF / black-hole render full-frame
        # (no gsplat overlay) instead of the live preview.
        if show_result.value and state["last_render"] is not None:
            t = torch.from_numpy(state["last_render"]).permute(2, 0, 1)[None].to(device)
            t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear",
                                                align_corners=False)
            return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)
        viewmat = torch.linalg.inv(c2w)[None]
        colors, _, _ = gsplat.rasterization(
            g.means, g.quats, g.scales, g.opacity, sh,
            viewmat, K[None], W, H, sh_degree=sh_degree, render_mode="RGB")
        img = colors[0].clamp(0, 1).cpu().numpy()
        bh = np.array([float(x) for x in gizmo.position], np.float32)
        Knp = K.cpu().numpy()
        pc = (np.linalg.inv(c2w.cpu().numpy()) @ np.array([*bh, 1.0], np.float32))[:3]
        if pc[2] > 1e-3 and state["r_s"] > 0:
            u = Knp[0, 0] * pc[0] / pc[2] + Knp[0, 2]
            v = Knp[1, 1] * pc[1] / pc[2] + Knp[1, 2]
            rad = Knp[1, 1] * (B_CRIT * state["r_s"]) / pc[2]
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
            img[np.abs(dist - rad) <= max(1.5, rad * 0.03)] = (1.0, 0.45, 0.1)
            img[dist <= 3.0] = (1.0, 0.2, 0.05)
        return img

    # capture the Rendering-panel's CameraPath instance (so we render its EXACT
    # KochanekBartels trajectory — no reinventing camera interpolation).
    import nerfview.render_panel as _rp
    _cam_path_ref = {}
    _orig_cp_init = _rp.CameraPath.__init__
    def _cp_init(self, *a, **k):
        _orig_cp_init(self, *a, **k)
        _cam_path_ref["path"] = self
    _rp.CameraPath.__init__ = _cp_init

    viewer = nerfview.Viewer(server, render_fn, mode="rendering")
    state["cam_path"] = _cam_path_ref.get("path")

    from nerfview._renderer import RenderTask
    def rerender():
        try:
            for cid, client in server.get_clients().items():
                viewer._renderers[cid].submit(RenderTask("move", viewer.get_camera_state(client)))
        except Exception as e:
            print("rerender skipped:", e, flush=True)

    @rs_slider.on_update
    def _(_):
        state["r_s"] = float(rs_slider.value); rerender()

    @str_slider.on_update
    def _(_):
        state["strength"] = float(str_slider.value)

    gizmo.on_update(lambda _: rerender())

    @server.on_client_connect
    def _(client):
        try:
            gizmo.position = np.array(client.camera.look_at, np.float32)
        except Exception as e:
            print("could not set BH to look_at:", e, flush=True)

        # moving the camera auto-unfreezes the result view, so the live preview
        # always matches the real camera (else a snapshot fires from a pose that
        # differs from the frozen image still on screen).
        @client.camera.on_update
        def _(_):
            if show_result.value:
                show_result.value = False

    # ---- helpers ---------------------------------------------------------------
    def cam_c2w_K(W, H):
        cs = state["last_cam"]                 # the camera render_fn is actually showing
        return np.asarray(cs.c2w, np.float32), np.asarray(cs.get_K((W, H)), np.float32)

    def to_u8(img):
        # model RGB is already in sRGB/display space (trained on sRGB images) — do NOT
        # gamma-correct again, or the saved PNG washes out / looks hazy vs the viewer.
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def curved_cfg():
        bh = tuple(float(x) for x in gizmo.position)
        return nf.CurvedConfig(
            bh_center=bh, rs=float(rs_slider.value), strength=float(str_slider.value),
            near=bh_near, far=bh_far, n_steps=int(steps_slider.value))

    # ---- snapshot: NeRF straight (rs=0) or curved (rs>0) -----------------------
    def do_snapshot():
        if state["last_cam"] is None:
            status.value = "move the camera once in the browser first"; return
        W = int(res_dd.value); H = int(W * 9 / 16)
        c2w, K = cam_c2w_K(W, H)
        rs = float(rs_slider.value)
        t0 = time.time()
        if rs > 0:
            status.value = f"black-hole render {W}x{H} rs={rs:.3f} steps={int(steps_slider.value)}…"
            img = nf.render_curved(nm, c2w, K, W, H, curved_cfg(), ss=int(ss_dd.value))
            tag = f"bh_rs{rs:.3f}"
        else:
            # NATIVE nerfacto via OpenCV RayBundle (proposal sampling, same frame as gsplat)
            status.value = f"NeRF native render {W}x{H}…"
            img = nf.render_straight(nm, c2w, K, W, H)
            tag = "native"
        u8 = to_u8(img)
        path = os.path.join(args.outdir, f"snapshot_{tag}.png")
        import imageio.v3 as iio
        iio.imwrite(path, u8)
        state["last_render"] = np.clip(img, 0, 1).astype(np.float32)
        show_result.value = True; rerender()
        status.value = f"saved {path}  ({time.time()-t0:.1f}s) — showing result"
        print(status.value, flush=True)

    # ---- compare: gsplat vs NeRF straight (alignment) --------------------------
    @torch.no_grad()
    def gsplat_render(c2w_np, K_np, W, H):
        c2w = torch.from_numpy(np.asarray(c2w_np, np.float32)).to(device)
        Kt = torch.from_numpy(np.asarray(K_np, np.float32)).to(device)
        colors, _, _ = gsplat.rasterization(
            g.means, g.quats, g.scales, g.opacity, sh,
            torch.linalg.inv(c2w)[None], Kt[None], W, H, sh_degree=sh_degree, render_mode="RGB")
        return colors[0].clamp(0, 1).cpu().numpy()

    def do_compare():
        if state["last_cam"] is None:
            status.value = "move the camera once in the browser first"; return
        W = int(res_dd.value); H = int(W * 9 / 16)
        c2w, K = cam_c2w_K(W, H)
        status.value = f"comparing gsplat vs NeRF {W}x{H}…"
        t0 = time.time()
        gimg = gsplat_render(c2w, K, W, H)
        nimg = nf.render_straight(nm, c2w, K, W, H)      # native nerfacto, OpenCV-aligned
        gu, nu = to_u8(gimg), to_u8(nimg)
        pad = 6
        sbs = np.full((H, W * 2 + pad, 3), 30, np.uint8)
        sbs[:, :W] = gu; sbs[:, W + pad:] = nu
        path = os.path.join(args.outdir, "compare_view.png")
        import imageio.v3 as iio
        iio.imwrite(path, sbs)
        state["last_render"] = np.clip(nimg, 0, 1).astype(np.float32)   # raw NeRF (frozen view)
        show_result.value = True; rerender()
        status.value = (f"saved {path}  gsplat {gimg.mean():.3f} | NeRF {nimg.mean():.3f} "
                        f"({time.time()-t0:.1f}s)")
        print(status.value, flush=True)

    # ---- native (proposal) vs OUR curved sampler at rs=0 (same sampler check) ---
    def do_native_vs_curved():
        if state["last_cam"] is None:
            status.value = "move the camera once in the browser first"; return
        W = int(res_dd.value); H = int(W * 9 / 16)
        c2w, K = cam_c2w_K(W, H)
        status.value = f"native vs curved(rs=0) {W}x{H}…"
        t0 = time.time()
        nat = nf.render_straight(nm, c2w, K, W, H)
        cfg = curved_cfg(); cfg.rs = 0.0                 # our sampler, no bending
        cur = nf.render_curved(nm, c2w, K, W, H, cfg, ss=int(ss_dd.value), force_curved=True)
        nu, cu = to_u8(nat), to_u8(cur)
        pad = 6
        sbs = np.full((H, W * 2 + pad, 3), 30, np.uint8)
        sbs[:, :W] = nu; sbs[:, W + pad:] = cu
        path = os.path.join(args.outdir, "compare_native_curved.png")
        import imageio.v3 as iio
        iio.imwrite(path, sbs)
        state["last_render"] = (sbs.astype(np.float32) / 255.0)
        show_result.value = True; rerender()
        status.value = (f"saved {path}  native {nat.mean():.3f} | curved(rs=0) {cur.mean():.3f} "
                        f"({time.time()-t0:.1f}s)")
        print(status.value, flush=True)

    # ---- black-hole orbit video ------------------------------------------------
    def do_video():
        if state["last_cam"] is None:
            status.value = "move the camera once in the browser first"; return
        W = int(res_dd.value); H = int(W * 9 / 16)
        ss = int(ss_dd.value)
        n = int(nframes.value); total = float(orbit_deg.value)
        cfg = curved_cfg()
        c2w0, K = cam_c2w_K(W, H)
        bh = np.array(cfg.bh_center, np.float32)
        up = up_pca if up_dropdown.value == "pca" else _axis_vec(up_dropdown.value)
        up = up / (np.linalg.norm(up) + 1e-9)
        rel = c2w0[:3, 3] - bh
        import imageio
        path = os.path.join(args.outdir, f"orbit_rs{cfg.rs:.3f}.mp4")
        writer = imageio.get_writer(path, fps=24, quality=8)
        t0 = time.time()
        for i in range(n):
            ang = math.radians(total * i / n)
            cam_i = bh + _rotate(rel, up, ang)
            c2w = _look_at_c2w(cam_i, bh, up)
            img = nf.render_curved(nm, c2w, K, W, H, cfg, ss=ss)
            writer.append_data(to_u8(img))
            progress.value = (i + 1) / n
            status.value = f"video {i+1}/{n}  ({time.time()-t0:.0f}s)"
        writer.close(); progress.value = 0.0
        status.value = f"saved {path}  ({n} frames, {time.time()-t0:.0f}s)"
        print(status.value, flush=True)

    def _run_bg(fn):
        if state["busy"]:
            status.value = "already rendering — wait"; return
        def wrap():
            state["busy"] = True
            try:
                fn()
            except Exception as e:
                status.value = f"error: {e}"
                import traceback; traceback.print_exc()
            finally:
                state["busy"] = False
        threading.Thread(target=wrap, daemon=True).start()

    snap_btn.on_click(lambda _: _run_bg(do_snapshot))
    cmp_btn.on_click(lambda _: _run_bg(do_compare))
    ncmp_btn.on_click(lambda _: _run_bg(do_native_vs_curved))
    vid_btn.on_click(lambda _: _run_bg(do_video))
    renderkf_btn.on_click(lambda _: _run_bg(do_kf_video))

    # ---- hook nerfview's Rendering-panel "Add Keyframe": capture camera + r_s ---
    rkfh = getattr(viewer, "_rendering_tab_handles", {})

    def _refresh_rskf():
        ks = state["rs_keyframes"]
        rskf_info.value = " → ".join(f"{k['rs']:.3f}" for k in ks) if ks else "(none)"

    def _capture_kf(_=None):
        # the Rendering panel records the CAMERA keyframe; we just record r_s for it.
        state["rs_keyframes"].append({"rs": float(rs_slider.value),
                                      "strength": float(str_slider.value),
                                      "bh": np.array(gizmo.position, np.float32)})
        _refresh_rskf()
        print(f"keyframe {len(state['rs_keyframes'])}: r_s={rs_slider.value:.3f}", flush=True)

    def _clear_kf(_=None):
        state["rs_keyframes"].clear(); _refresh_rskf()

    try:
        rkfh["add_keyframe_button"].on_click(_capture_kf)
        rkfh["clear_keyframes_button"].on_click(_clear_kf)
        print("hooked Rendering-panel Add/Clear Keyframe", flush=True)
    except Exception as e:
        print("could not hook Rendering panel:", e, flush=True)

    def _lerp_kf(t):
        ks = state["rs_keyframes"]; n = len(ks)
        pos = max(0.0, min(1.0, t)) * (n - 1)
        a = int(pos); b = min(a + 1, n - 1); f = pos - a
        rs = ks[a]["rs"] * (1 - f) + ks[b]["rs"] * f
        stg = ks[a]["strength"] * (1 - f) + ks[b]["strength"] * f
        bh = ks[a]["bh"] * (1 - f) + ks[b]["bh"] * f
        return rs, stg, bh

    def do_kf_video():
        # Pre-compute every frame's camera (from the Rendering-panel spline) + r_s,
        # save to .npz, then launch a DETACHED nohup render job that survives viewer
        # restarts / SSH drop. (The render itself runs outside this process.)
        path = _cam_path_ref.get("path")
        ks = state["rs_keyframes"]
        if path is None or len(getattr(path, "_keyframes", {})) < 2 or len(ks) < 2:
            status.value = "add >=2 keyframes in the Rendering panel"; return
        try:
            path.update_spline()
        except Exception:
            pass
        W = int(res_dd.value); H = int(W * 9 / 16); ss = int(ss_dd.value)
        try:
            fps = float(rkfh["framerate_number"].value)
            total = max(2, int(fps * float(rkfh["duration_number"].value)))
        except Exception:
            fps, total = 24.0, 120
        C2W = np.zeros((total, 4, 4), np.float32); Km = np.zeros((total, 3, 3), np.float32)
        RS = np.zeros(total, np.float32); STG = np.zeros(total, np.float32)
        BH = np.zeros((total, 3), np.float32)
        for i in range(total):
            t = i / (total - 1)
            res = path.interpolate_pose_and_fov_rad(t)
            if res is None:
                status.value = "spline not ready — re-add keyframes"; return
            pose, fov = res[0], res[1]
            C2W[i] = np.asarray(pose.as_matrix(), np.float32)
            fy = H / (2.0 * math.tan(fov * 0.5))
            Km[i] = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)
            rs, stg, _ = _lerp_kf(t)
            # BH position = the gizmo's CURRENT position (where you last moved it),
            # fixed for the whole video — NOT the per-keyframe captured position.
            RS[i] = rs; STG[i] = stg; BH[i] = np.array(gizmo.position, np.float32)
        job = os.path.join(args.outdir, "video_job.npz")
        np.savez(job, c2w=C2W, K=Km, rs=RS, strength=STG, bh=BH, W=W, H=H, fps=fps,
                 near=bh_near, far=bh_far, n_steps=int(steps_slider.value))
        out = os.path.join(args.outdir, "blackhole_video.mp4")
        here = os.path.dirname(os.path.abspath(__file__))
        log = "/tmp/video_job.log"
        cmd = [sys.executable, os.path.join(here, "render_video_job.py"),
               "--config", args.config, "--frames", job, "--out", out, "--ss", str(ss)]
        import subprocess
        env = dict(os.environ, CUDA_HOME="/home/ubuntu/miniconda3/envs/3DGS")
        with open(log, "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env,
                                 start_new_session=True)  # detach: survives this process
        status.value = (f"launched detached render: {total} frames -> {out}  "
                        f"(pid {p.pid}, log {log}) — survives viewer restart")
        print(status.value, flush=True)

    print("viewer ready on port", args.port, flush=True)
    while True:
        time.sleep(1.0)


def _axis_vec(name):
    m = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0],
         "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}
    return np.array(m.get(name, [0, 1, 0]), np.float32)


def _rotate(v, axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _look_at_c2w(eye, target, up):
    eye = np.asarray(eye, np.float32); target = np.asarray(target, np.float32)
    fwd = target - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    down = np.cross(fwd, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = eye
    return c2w


if __name__ == "__main__":
    main()
