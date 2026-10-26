# Lunar habitat assembly: robot learning in 1/6 g

A rover-mounted robot arm builds a habitat on the Moon. It picks a sintered-regolith wall panel off
its deck and seats it into the gap in a half-built habitat ring, under lunar gravity (1.62 m/s²).
The policy is trained end to end with reinforcement learning on a CPU.

The robot is NVIDIA's open SimReady Franka Panda from
[`nvidia/isaacsim-assets`](https://huggingface.co/datasets/nvidia/isaacsim-assets) on Hugging Face. It is
converted from OpenUSD to MuJoCo by a converter in this repo, so it runs without Isaac Sim or an
NVIDIA GPU.

## Why the Moon changes the problem

On Earth, gravity does a lot of quiet work in manipulation: dropped objects land and settle, and
contact forces are large enough that friction stops things sliding. At 1/6 g:

- A 1.2 kg panel weighs 1.9 N instead of 11.8 N. Its inertia is unchanged, so swinging and
  momentum dominate while it is carried.
- A panel released a few centimetres high takes about 2.4× longer to fall and lands with
  little settling force, so it rocks and walks out of the socket.
- Normal forces in contact are 6× smaller, and friction drops with them.

The habitat socket is a U-channel with 7 mm clearance. A placement succeeds only when the panel
sits upright within 2 cm, has been released, and has come to rest.

## Pipeline

```
NVIDIA SimReady USD ──tools/usd_to_mjcf.py──▶ MJCF robot ──scene.py──▶ lunar site ──env.py──▶ Gymnasium task ──train.py──▶ SAC policy
```

| Part | File | What it does |
|---|---|---|
| Asset fetch | `tools/fetch_assets.py` | Downloads selected robot packages from the Hugging Face dataset |
| USD → MJCF | `tools/usd_to_mjcf.py` | Reads UsdPhysics bodies, joints, limits, drives, mass properties and mimic fingers. Rebuilds the zero-pose kinematic tree from the joint frames (NVIDIA authors the robot in a bent pose). Bakes meshes per body. Works for any SimReady arm in the dataset. |
| Scene | `src/lunar_habitat/scene.py` | Procedural cratered regolith heightfield, rover, habitat ring with a U-channel socket, Earth on the horizon, low raking sun |
| Task | `src/lunar_habitat/env.py` | 20 Hz Cartesian control through damped-least-squares IK, staged dense reward, strict success check |
| Baseline | `scripts/scripted_demo.py` | Hand-written controller, used to validate the task and to seed the replay buffer |
| Training | `scripts/train.py` | SAC with 10 parallel environments, a demo-seeded replay buffer and periodic success-rate evaluation |
| Video | `scripts/render_policy.py` | Renders a policy from the cinematic cameras |

**Converter check:** at the zero pose, the converted Franka's hand flange sits at
(0.088, 0, 0.926) m. That matches Franka's published kinematics exactly, and the joint limits
and 18 kg mass match the datasheet.

## Results

100 held-out episodes per cell, with seeds disjoint from training. Success means the panel is seated
upright within 2 cm, released, and at rest.

| Controller | Moon (1.62 m/s²) | Earth (9.81 m/s²) | Median time to place |
|---|---|---|---|
| **SAC policy** (trained on the Moon) | **81%** | 2% | 6.5 s |
| Scripted baseline | 42% | 1% | 8.8 s |

Successful placements land with a median 7 mm position error.

- **Training:** 600k steps from scratch, then 100k steps of fine-tuning after a reward fix (below).
  This took about 1.5 hours on a 16-core CPU, with 10 parallel environments and the replay buffer
  seeded with 200 noisy scripted episodes.
- **Reward hacking, found and fixed:** at 600k steps the policy seated and released panels but
  never moved away, so it never counted as a success. Ending the episode paid +10, while hovering
  beside a placed panel paid 1.0 per step, worth about 50 at γ = 0.98. Raising the terminal bonus
  to 75 took evaluation success from 0% to 90% within 100k steps.
- **Earth gravity:** both controllers fail at 9.81 m/s². The arm tracks worse under the 6× heavier
  load, and the policy never saw that regime. A gravity-randomised policy is the next experiment
  (`--gravity 1.0 9.81`).

## Run it

```bash
source scripts/env.sh                      # shared uv tooling from ~/robotics-tooling
uv sync
uv run python tools/fetch_assets.py frankarobotics_frankapanda_franka
uv run python tools/usd_to_mjcf.py assets/nvidia/frankarobotics_frankapanda_franka/franka.usda \
    --variant MuJoCo=Enabled --out assets/converted/franka --name franka
uv run python scripts/scripted_demo.py --episodes 20                 # baseline
uv run python scripts/train.py --steps 3000000 --demos 200 --run moon_sac
uv run python scripts/evaluate.py runs/moon_sac/best_model.zip --episodes 100 --gravity 1.62 9.81
MUJOCO_GL=egl uv run python scripts/render_policy.py runs/moon_sac/best_model.zip --out videos/moon.mp4
```

Hardware used: 16-core CPU with an AMD RX 580. No CUDA is needed; rendering uses EGL.

## Limits

- Policies act on simulator state (poses and velocities), not camera images.
- Panels are rigid boxes, and regolith dust is not simulated.
- The real Franka has 7 kg payload and joint torque limits that are respected here. Its
  controller is idealised (position actuators plus IK, no torque-sensing compliance).

## Licences

NVIDIA SimReady assets are Apache-2.0 (see `assets/nvidia/*/.metadata/license.txt`). They are
downloaded, not redistributed.
