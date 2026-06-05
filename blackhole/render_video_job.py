"""render_video_job.py — standalone black-hole video renderer (detached job).

The viewer pre-computes every frame's camera (c2w, K) and black-hole params
(rs, strength, bh) from the Rendering-panel spline, saves them to an .npz, then
launches THIS script with nohup so the render survives viewer restarts / SSH drop.

    python render_video_job.py --config <nerfacto config.yml> \
        --frames renders/video_job.npz --out renders/blackhole_video.mp4 --ss 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nerf_renderer as nf          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", required=True, help="npz with per-frame params")
    ap.add_argument("--out", default="renders/blackhole_video.mp4")
    ap.add_argument("--ss", type=int, default=1)
    ap.add_argument("--width", type=int, default=0, help="override render width (rescales K)")
    args = ap.parse_args()

    nm = nf.load_nerf(args.config)
    d = np.load(args.frames)
    c2w = d["c2w"].astype(np.float32)      # [N,4,4] (resolution-independent camera path)
    K = d["K"].astype(np.float32)          # [N,3,3]
    rs = d["rs"]; stg = d["strength"]; bh = d["bh"]
    W, H = int(d["W"]), int(d["H"])
    if args.width and args.width != W:
        Wn = int(args.width); Hn = int(Wn * H / W)        # keep aspect
        sc = Hn / H
        K = K.copy(); K[:, 0, 0] *= sc; K[:, 1, 1] *= sc
        K[:, 0, 2] = Wn / 2.0; K[:, 1, 2] = Hn / 2.0
        W, H = Wn, Hn
    fps = float(d["fps"]); near = float(d["near"]); far = float(d["far"])
    n_steps = int(d["n_steps"])
    N = c2w.shape[0]
    print(f"[job] {N} frames {W}x{H} ss{args.ss} fps{fps:.0f} -> {args.out}", flush=True)

    import imageio
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=int(fps), quality=8)
    t0 = time.time()
    for i in range(N):
        cfg = nf.CurvedConfig(bh_center=tuple(float(x) for x in bh[i]),
                              rs=float(rs[i]), strength=float(stg[i]),
                              near=near, far=far, n_steps=n_steps)
        img = nf.render_curved(nm, c2w[i], K[i], W, H, cfg, ss=args.ss)
        writer.append_data((np.clip(img, 0, 1) * 255).astype(np.uint8))
        print(f"[job] frame {i+1}/{N}  rs={float(rs[i]):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    writer.close()
    print(f"[job] DONE {args.out}  ({N} frames, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
