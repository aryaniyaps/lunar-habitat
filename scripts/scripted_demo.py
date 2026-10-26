"""Hand-written controller for the panel task. Used to validate the environment (the task is
solvable, the success check fires) and as a baseline for the learned policy.

    python scripts/scripted_demo.py [--gravity 1.62] [--episodes 10] [--video out.mp4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunar_habitat.env import LunarPanelEnv  # noqa: E402


def run_episode(env, frames=None, rng=None):
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)) if rng is not None else None)
    info = {}

    def go(target_fn, grip, n, gain=10, tol=None):
        nonlocal info
        for _ in range(n):
            a = np.clip((target_fn() - env.tcp()) * gain, -1, 1)
            _, _, term, trunc, info = env.step([*a, grip])
            if frames is not None:
                frames.append(env.render())
            if term or trunc:
                return True
            if tol and np.linalg.norm(target_fn() - env.tcp()) < tol:
                return False
        return False

    gp = env.grasp_point
    if go(lambda: gp() + [0, 0, 0.10], 1, 40, tol=0.01): return info
    if go(gp, 1, 30, gain=6, tol=0.004): return info
    if go(env.tcp, -1, 8): return info
    off = env.tcp() - env.data.xpos[env.panel_id]
    if go(lambda: env.tcp() * [1, 1, 0] + [0, 0, 0.62], -1, 30, tol=0.02): return info
    if go(lambda: env.target_pos + off + [0, 0, 0.24], -1, 60, tol=0.01): return info
    if go(lambda: env.target_pos + off + [0, 0, 0.002], -1, 60, gain=5, tol=0.004): return info
    if go(env.tcp, 1, 8): return info
    go(lambda: env.target_pos + off + [0, 0, 0.2], 1, 40)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravity", type=float, default=1.62)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()
    env = LunarPanelEnv(gravity=args.gravity, render_mode="rgb_array" if args.video else None)
    rng = np.random.default_rng(0)
    frames = [] if args.video else None
    wins = 0
    for ep in range(args.episodes):
        info = run_episode(env, frames, rng)
        wins += bool(info.get("success"))
        print(f"ep {ep}: success={info.get('success')} xy={info.get('xy_err', 0):.3f} z={info.get('z_err', 0):.3f} tilt={info.get('tilt_deg', 0):.1f}")
    print(f"success rate {wins}/{args.episodes} at g={args.gravity}")
    if args.video:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=20, quality=8)


if __name__ == "__main__":
    main()
