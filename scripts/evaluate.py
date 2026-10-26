"""Evaluate controllers on held-out seeds at several gravity levels.

    python scripts/evaluate.py runs/moon_sac_v2/best_model.zip --episodes 100 --gravity 1.62 9.81
Writes a JSON summary next to the model (or to --out).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lunar_habitat.env import CTRL_HZ, LunarPanelEnv  # noqa: E402
from scripted_demo import run_episode  # noqa: E402

EVAL_SEED = 2_000_000  # disjoint from training (1000+) and in-training eval (10000+) seeds


def eval_policy(policy, gravity, episodes):
    env = LunarPanelEnv(gravity=gravity)
    wins, times, errs, tilts = 0, [], [], []
    for ep in range(episodes):
        if policy == "scripted":
            info = run_episode(env, rng=np.random.default_rng(EVAL_SEED + ep))
            steps = env._steps
        else:
            obs, _ = env.reset(seed=EVAL_SEED + ep)
            done = False
            while not done:
                a, _ = policy.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            steps = env._steps
        if info.get("success"):
            wins += 1
            times.append(steps / CTRL_HZ)
            errs.append(info["xy_err"] * 1000)
            tilts.append(info["tilt_deg"])
    return {
        "gravity": gravity, "episodes": episodes, "successes": wins, "success_rate": wins / episodes,
        "median_time_s": float(np.median(times)) if times else None,
        "median_xy_error_mm": float(np.median(errs)) if errs else None,
        "median_tilt_deg": float(np.median(tilts)) if tilts else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="SAC .zip or 'scripted'")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--gravity", type=float, nargs="+", default=[1.62, 9.81])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.policy == "scripted":
        policy, out = "scripted", args.out or ROOT / "runs" / "scripted_eval.json"
    else:
        from stable_baselines3 import SAC
        policy = SAC.load(args.policy, device="cpu")
        out = args.out or Path(args.policy).with_suffix(".eval.json")
    results = []
    for g in args.gravity:
        t = time.time()
        r = eval_policy(policy, g, args.episodes)
        r["wall_s"] = round(time.time() - t, 1)
        results.append(r)
        print(json.dumps(r), flush=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
