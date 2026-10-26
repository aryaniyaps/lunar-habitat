"""Render a trained policy (or the scripted baseline) to an mp4 from the cinematic camera.

    python scripts/render_policy.py runs/moon_sac_v1/best_model.zip --out videos/moon.mp4 --episodes 3
    python scripts/render_policy.py scripted --out videos/scripted.mp4
"""
import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lunar_habitat.env import LunarPanelEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="path to a SAC .zip, or 'scripted'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--gravity", type=float, default=1.62)
    ap.add_argument("--camera", default="hero")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = LunarPanelEnv(gravity=args.gravity, render_mode="rgb_array", camera=args.camera, cinematic=True)
    frames, results = [], []
    rng = np.random.default_rng(args.seed)
    if args.policy == "scripted":
        from scripted_demo import run_episode
        for _ in range(args.episodes):
            info = run_episode(env, frames, rng)
            results.append(bool(info.get("success")))
            frames.extend([frames[-1]] * 20)
    else:
        from stable_baselines3 import SAC
        model = SAC.load(args.policy, device="cpu")
        for _ in range(args.episodes):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                frames.append(env.render())
                done = term or trunc
            results.append(bool(info["success"]))
            frames.extend([frames[-1]] * 20)  # hold the last frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=20, quality=8, macro_block_size=8)
    print(f"{sum(results)}/{len(results)} successful -> {args.out}")


if __name__ == "__main__":
    main()
