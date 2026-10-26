"""Render stills of the scene from each camera: python scripts/snapshot.py out_dir"""
import sys
from pathlib import Path

import imageio
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunar_habitat.scene import SceneConfig, build  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "videos")
out.mkdir(parents=True, exist_ok=True)
m = build(SceneConfig(cinematic=True))
d = mujoco.MjData(m)
d.qpos[:7] = [0, -0.3, 0, -2.2, 0, 1.9, 0.785]
d.qpos[7:9] = 0.04
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 720, 1280)
for cam in ["hero", "wide", "top"]:
    r.update_scene(d, camera=cam)
    imageio.imwrite(out / f"{cam}.png", r.render())
print("saved", out)
