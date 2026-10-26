"""Train an end-to-end policy for the lunar panel task with SAC.

    python scripts/train.py --steps 3000000 --envs 10 --gravity 1.62 --run moon_sac

Logs to runs/<run>/ (TensorBoard, checkpoints, eval success rate).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunar_habitat.env import LunarPanelEnv  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from scripted_demo import run_episode  # noqa: E402


def make_env(gravity, seed):
    def _f():
        return LunarPanelEnv(gravity=gravity, seed=seed)
    return _f


class EvalSuccess(BaseCallback):
    """Every `every` steps, run deterministic episodes and log the success rate."""

    def __init__(self, gravity, every, episodes, out_dir):
        super().__init__()
        self.gravity, self.every, self.episodes, self.out_dir = gravity, every, episodes, out_dir
        self.next_eval = every
        self.env = LunarPanelEnv(gravity=gravity, seed=12345)
        self.best = -1.0
        self.history = []

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.every
        wins, stages = 0, []
        for ep in range(self.episodes):
            obs, _ = self.env.reset(seed=10_000 + ep)
            done = False
            while not done:
                a, _ = self.model.predict(obs[None], deterministic=True)
                obs, r, term, trunc, info = self.env.step(a[0])
                done = term or trunc
            wins += info["success"]
            stages.append(self.env.stage_best)
        rate = wins / self.episodes
        self.logger.record("eval/success_rate", rate)
        self.logger.record("eval/best_stage", float(np.mean(stages)))
        self.history.append({"steps": self.num_timesteps, "success_rate": rate, "mean_best_stage": float(np.mean(stages)),
                             "wall_s": time.time() - self.t0})
        (self.out_dir / "eval.json").write_text(json.dumps(self.history, indent=1))
        print(f"[eval] steps={self.num_timesteps} success={rate:.2f} stage={np.mean(stages):.2f}", flush=True)
        if rate >= self.best:
            self.best = rate
            self.model.save(self.out_dir / "best_model")
        return True

    def _on_training_start(self):
        self.t0 = time.time()


def seed_replay_buffer(model, gravity, episodes, n_envs):
    """Fill the replay buffer with noisy scripted-controller episodes (successful or not)."""
    env = LunarPanelEnv(gravity=gravity, seed=777)
    rng = np.random.default_rng(777)
    trans, wins = [], 0
    for ep in range(episodes):
        info = run_episode(env, rng=rng, noise=0.15 if ep % 2 else 0.0,
                           on_step=lambda o, a, r, no, te, tr, inf: trans.append((o, a, r, no, te, inf)))
        wins += bool(info.get("success"))
    # the buffer stores n_envs transitions per add(); feed it in chunks
    for i in range(0, len(trans) - n_envs + 1, n_envs):
        chunk = trans[i:i + n_envs]
        model.replay_buffer.add(
            np.stack([c[0] for c in chunk]), np.stack([c[3] for c in chunk]),
            np.stack([c[1] for c in chunk]).clip(-0.999, 0.999), np.array([c[2] for c in chunk]),
            np.array([c[4] for c in chunk]), [dict(c[5], **{"TimeLimit.truncated": False}) for c in chunk])
    print(f"[demos] {episodes} scripted episodes, {wins} successful, {len(trans)} transitions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--gravity", type=float, nargs="+", default=[1.62], help="one value, or LOW HIGH to randomise")
    ap.add_argument("--run", default="moon_sac")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--demos", type=int, default=0, help="scripted episodes to seed the replay buffer with")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    gravity = args.gravity[0] if len(args.gravity) == 1 else tuple(args.gravity)
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))

    venv = SubprocVecEnv([make_env(gravity, 1000 + i) for i in range(args.envs)])
    venv = VecMonitor(venv)
    model = SAC(
        "MlpPolicy", venv, learning_rate=3e-4, buffer_size=2_000_000, batch_size=512, gamma=0.98, tau=0.005,
        train_freq=1, gradient_steps=max(1, args.envs // 4), learning_starts=20_000, ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256, 256]), tensorboard_log=str(out / "tb"), verbose=0, seed=0,
    )
    if args.resume:
        model.set_parameters(args.resume)
    if args.demos:
        seed_replay_buffer(model, gravity, args.demos, args.envs)
    eval_g = gravity if isinstance(gravity, float) else 1.62
    callbacks = [
        CheckpointCallback(save_freq=max(1, 250_000 // args.envs), save_path=str(out / "ckpt")),
        EvalSuccess(eval_g, args.eval_every, 20, out),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=10, progress_bar=False)
    model.save(out / "final_model")


if __name__ == "__main__":
    main()
