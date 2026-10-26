"""Gymnasium task: pick a regolith panel off the rover and set it into the habitat ring gap.

Action (4,) in [-1, 1]: Cartesian velocity of the gripper TCP (x, y, z) and a gripper command
(> 0 open, < 0 close). A damped-least-squares IK layer turns the TCP target into joint position
targets for the Franka's position actuators; TCP orientation is held top-down with the fingers
closing along world x (across the panel's thickness).

Observation (flat, 32,): TCP pos/vel, finger opening, panel pose (pos + 6D rotation) and
velocities, panel-to-TCP and panel-to-target vectors, target position, gravity magnitude.
Including gravity lets one policy be trained across gravity levels (domain randomisation).
"""
from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np

from .scene import DECK_Z, EARTH_G, MOON_G, PANEL, SceneConfig, build, slot_pose

CTRL_HZ = 20
SIM_DT = 0.002
SUBSTEPS = int(round(1 / (CTRL_HZ * SIM_DT)))
MAX_SPEED = 0.5            # m/s TCP speed at |a| = 1
WORKSPACE_LO = np.array([0.05, -0.45, 0.16])
WORKSPACE_HI = np.array([0.85, 0.35, 0.95])
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, -0.785])
TCP_OFFSET = 0.1034        # flange to fingertip centre along hand z
SUCCESS_XY = 0.02
SUCCESS_Z = 0.015
SUCCESS_TILT = math.radians(8)
# Success ends the episode, so its bonus must beat hovering beside a placed panel forever
# (max stage 1.0 per step is worth 1 / (1 - gamma) = 50 at gamma = 0.98).
SUCCESS_BONUS = 75.0


class LunarPanelEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, gravity: float | tuple[float, float] = MOON_G, horizon: int = 300, seed: int | None = None,
                 render_mode: str | None = None, camera: str = "hero", width: int = 1280, height: int = 720,
                 cinematic: bool = False):
        self.gravity_spec = gravity
        self.horizon = horizon
        self.render_mode = render_mode
        self.camera = camera
        self.model = build(SceneConfig(gravity=MOON_G, cinematic=cinematic))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"panda_joint{i}").id] for i in range(1, 8)])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(f"panda_joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(f"panda_finger_joint{i}").id] for i in (1, 2)])
        self.jnt_lo = m.jnt_range[[m.joint(f"panda_joint{i}").id for i in range(1, 8)], 0]
        self.jnt_hi = m.jnt_range[[m.joint(f"panda_joint{i}").id for i in range(1, 8)], 1]
        self.hand_id = m.body("panda_hand").id
        self.panel_id = m.body("panel").id
        self.panel_qadr = m.jnt_qposadr[m.joint("panel_free").id]
        self.panel_dadr = m.jnt_dofadr[m.joint("panel_free").id]
        self.panel_geoms = {m.geom("panel").id}
        self.finger_geoms = {i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.endswith("finger")
                             and m.geom_contype[i]}
        self.target_pos, target_yaw = slot_pose(12)
        self.target_yaw = target_yaw
        self.panel_start = self.model.body_pos[self.panel_id].copy()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (32,), np.float32)
        self._renderer = None
        self._jac_p = np.zeros((3, m.nv))
        self._jac_r = np.zeros((3, m.nv))
        self._init_seed = seed
        self._steps = 0

    # ------------------------------------------------------------------ helpers
    def tcp(self) -> np.ndarray:
        d = self.data
        R = d.xmat[self.hand_id].reshape(3, 3)
        return d.xpos[self.hand_id] + R[:, 2] * TCP_OFFSET

    def finger_width(self) -> float:
        return float(self.data.qpos[self.finger_qadr].sum())

    def panel_R(self) -> np.ndarray:
        return self.data.xmat[self.panel_id].reshape(3, 3)

    def panel_tilt(self) -> float:
        return float(math.acos(np.clip(self.panel_R()[2, 2], -1, 1)))

    def grasp_point(self) -> np.ndarray:
        # fingertips 2.8 cm below the top edge (the palm hull starts 3.6 cm above the TCP), centred on the grip strip
        return self.data.xpos[self.panel_id] + self.panel_R()[:, 2] * (PANEL[2] - 0.028)

    def _finger_contacts(self) -> tuple[bool, bool]:
        left = right = False
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self.panel_geoms:
                other = (pair - self.panel_geoms) or pair
                for g in other:
                    name = m.body(m.geom_bodyid[g]).name
                    left |= name == "panda_leftfinger"
                    right |= name == "panda_rightfinger"
        return left, right

    def grasped(self) -> bool:
        left, right = self._finger_contacts()
        return left and right

    def _ik(self, tcp_target: np.ndarray) -> np.ndarray:
        """One damped-least-squares step from the current arm targets toward tcp_target."""
        m, d = self.model, self.data
        R = d.xmat[self.hand_id].reshape(3, 3)
        tcp = self.tcp()
        mujoco.mj_jac(m, d, self._jac_p, self._jac_r, tcp, self.hand_id)
        Jp = self._jac_p[:, self.arm_dadr]
        Jr = self._jac_r[:, self.arm_dadr]
        err_p = tcp_target - tcp
        # orientation error: R_des * R^T -> axis-angle
        Re = self.R_des @ R.T
        err_r = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        J = np.vstack([Jp, Jr])
        e = np.concatenate([err_p, err_r])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        # null-space pull toward home posture keeps the elbow sensible
        N = np.eye(7) - np.linalg.pinv(J) @ J
        q = d.qpos[self.arm_qadr]
        dq += N @ (0.2 * (HOME_Q - q))
        return np.clip(q + dq, self.jnt_lo + 0.02, self.jnt_hi - 0.02)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed if seed is not None else self._init_seed)
        self._init_seed = None
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if isinstance(self.gravity_spec, tuple):
            g = float(self.np_random.uniform(*self.gravity_spec))
        else:
            g = float(self.gravity_spec)
        if options and "gravity" in options:
            g = float(options["gravity"])
        m.opt.gravity[:] = (0, 0, -g)
        self.g = g
        q = HOME_Q + self.np_random.uniform(-0.05, 0.05, 7)
        d.qpos[self.arm_qadr] = q
        d.qpos[self.finger_qadr] = 0.04
        jitter = np.array([*self.np_random.uniform(-0.04, 0.04, 2), 0.0])
        d.qpos[self.panel_qadr:self.panel_qadr + 3] = self.panel_start + jitter
        yaw = self.np_random.uniform(-0.03, 0.03)  # the rack holds panels nearly aligned
        d.qpos[self.panel_qadr + 3:self.panel_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.ctrl[:7] = q
        d.ctrl[7] = 0.04
        mujoco.mj_forward(m, d)
        # desired hand orientation: pointing down, fingers closing along world x
        self.R_des = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        for _ in range(60):  # settle into the reference orientation before the episode starts
            d.ctrl[:7] = self._ik(self.tcp())
            mujoco.mj_step(m, d, SUBSTEPS // 5)
        d.qvel[:] = 0
        self.tcp_target = self.tcp().copy()
        self._steps = 0
        self._prev_place_dist = None
        self.stage_best = 0.0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.tcp_target = np.clip(self.tcp_target + a[:3] * MAX_SPEED / CTRL_HZ, WORKSPACE_LO, WORKSPACE_HI)
        # keep the target leash short so the policy cannot wind up a large error
        lag = self.tcp_target - self.tcp()
        if np.linalg.norm(lag) > 0.08:
            self.tcp_target = self.tcp() + lag / np.linalg.norm(lag) * 0.08
        d = self.data
        d.ctrl[7] = 0.04 if a[3] > 0 else 0.0
        for k in range(SUBSTEPS):
            if k % 5 == 0:
                d.ctrl[:7] = self._ik(self.tcp_target)
            mujoco.mj_step(self.model, d)
        self._steps += 1
        reward, info = self._reward()
        terminated = bool(info["success"])
        fell = d.xpos[self.panel_id][2] < 0.05 and not info["placed"]
        truncated = self._steps >= self.horizon or fell
        info["fell"] = bool(fell)
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ obs / reward
    def _obs(self) -> np.ndarray:
        d = self.data
        tcp = self.tcp()
        R = self.panel_R()
        hv = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.hand_id, hv, 0)
        pv = d.qvel[self.panel_dadr:self.panel_dadr + 6]
        ppos = d.xpos[self.panel_id]
        obs = np.concatenate([
            tcp, hv[3:], [self.finger_width()],
            ppos, R[:, 0], R[:, 2], pv,
            self.grasp_point() - tcp, self.target_pos - ppos, self.target_pos,
            [self.g / EARTH_G],
        ])
        return obs.astype(np.float32)

    def _reward(self) -> tuple[float, dict]:
        d = self.data
        tcp = self.tcp()
        ppos = d.xpos[self.panel_id]
        grasp_d = np.linalg.norm(self.grasp_point() - tcp)
        grasped = self.grasped()
        to_target = self.target_pos - ppos
        xy_d = np.linalg.norm(to_target[:2])
        z_d = abs(to_target[2])
        tilt = self.panel_tilt()
        placed = xy_d < SUCCESS_XY and z_d < SUCCESS_Z and tilt < SUCCESS_TILT
        released = placed and not any(self._finger_contacts()) and tcp[2] > ppos[2] + PANEL[2] + 0.04
        success = placed and released and np.linalg.norm(d.qvel[self.panel_dadr:self.panel_dadr + 6]) < 0.05

        # staged shaping: reach -> grasp -> carry above the gap -> lower into the gap -> release
        r_reach = 1 - math.tanh(8 * grasp_d)
        above = self.target_pos + np.array([0, 0, 0.24])  # panel bottom clears the 30 cm wall
        r_carry = 1 - math.tanh(5 * np.linalg.norm(above - ppos))
        r_insert = 1 - math.tanh(15 * (xy_d + z_d))
        lifted = ppos[2] > self.panel_start[2] + 0.03
        stage = 0.25 * r_reach
        if grasped:
            stage = max(stage, 0.35 + 0.15 * float(lifted) + 0.25 * r_carry * float(lifted))
            if xy_d < 0.04:
                stage = max(stage, 0.75 + 0.15 * r_insert)
        if placed:
            stage = max(stage, 0.92 + 0.08 * float(not grasped))
        upright_pen = 0.1 * min(tilt, 1.0) if lifted or placed else 0.0
        reward = stage - upright_pen + (SUCCESS_BONUS if success else 0.0)
        self.stage_best = max(self.stage_best, stage)
        info = dict(success=bool(success), placed=bool(placed), grasped=bool(grasped), stage=stage,
                    xy_err=float(xy_d), z_err=float(z_d), tilt_deg=math.degrees(tilt), gravity=self.g)
        return float(reward), info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 720, 1280)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register("LunarPanel-v0", entry_point="lunar_habitat.env:LunarPanelEnv")
