"""Procedural lunar construction site built around the converted NVIDIA Franka.

Layout (metres, z up):
  - Rover deck at the origin; the arm base sits on it at height DECK_Z.
  - A panel rack on the deck holds the next habitat panel upright.
  - The habitat ring (radius RING_R, centred at RING_CENTER) is partly built:
    panels stand on a regolith foundation; one gap within reach is the target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRANKA_XML = ROOT / "assets" / "converted" / "franka" / "franka.xml"

MOON_G = 1.62
EARTH_G = 9.81

DECK_Z = 0.30          # top of rover deck
PANEL = np.array([0.02, 0.11, 0.09])    # half-sizes: thickness, width, height (4 x 22 x 18 cm)
PANEL_MASS = 1.2       # kg (sintered regolith tile)
RING_R = 1.05          # habitat ring radius
RING_CENTER = np.array([1.75, 0.0])
FOUNDATION_Z = 0.12    # top of the foundation the panels stand on
N_SLOTS = 24           # panels around the ring


@dataclass
class SceneConfig:
    gravity: float = MOON_G
    seed: int = 0
    gap_slot: int = 12          # slot index that is left open as the target
    terrain_res: int = 192
    cinematic: bool = False     # extra visual-only props (dust, beacons)


def slot_pose(k: int) -> tuple[np.ndarray, float]:
    """World position (centre) and yaw of ring slot k. Slot 12 faces the robot."""
    a = 2 * math.pi * k / N_SLOTS
    x = RING_CENTER[0] + RING_R * math.cos(a)
    y = RING_CENTER[1] + RING_R * math.sin(a)
    return np.array([x, y, FOUNDATION_Z + PANEL[2]]), a  # panel normal points radially


def _terrain(res: int, seed: int) -> np.ndarray:
    """Heightfield in [0, 1]: rolling regolith with craters, flattened near the work site."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(xs, xs)
    h = np.zeros_like(X)
    for octave in range(5):
        f = 2 ** octave * 1.5
        phase = rng.uniform(0, 2 * np.pi, 4)
        h += (np.sin(f * np.pi * X + phase[0]) * np.cos(f * np.pi * Y + phase[1])
              + np.sin(f * 1.7 * np.pi * (X + Y) + phase[2])) / (2 ** octave)
    h = (h - h.min()) / (h.max() - h.min()) * 0.25
    for _ in range(26):
        cx, cy = rng.uniform(-1, 1, 2)
        r = rng.uniform(0.03, 0.18)
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r
        bowl = np.where(d < 1, -(1 - d ** 2), 0.0)
        rim = np.where((d >= 0.8) & (d < 1.4), np.exp(-((d - 1.05) ** 2) / 0.02) * 0.35, 0.0)
        h += (bowl + rim) * r * 1.6
    # Flatten the construction pad (terrain spans 40 m; pad ~6 m around the site).
    d_site = np.sqrt((X - 0.06) ** 2 + Y ** 2)
    w = np.clip((d_site - 0.18) / 0.15, 0, 1)
    h = h * w + h.mean() * 0.2 * (1 - w)
    h = (h - h.min()) / (h.max() - h.min())
    return h.astype(np.float32)


def _textures(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.0, 0.0, 0.01], rgb2=[0.0, 0.0, 0.0], width=512, height=3072,
                     mark=mujoco.mjtMark.mjMARK_RANDOM, markrgb=[1, 1, 1], random=0.0015)
    spec.add_texture(name="regolith", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.42, 0.41, 0.40], rgb2=[0.30, 0.29, 0.28], width=512, height=512,
                     mark=mujoco.mjtMark.mjMARK_RANDOM, markrgb=[0.55, 0.54, 0.52], random=0.25)
    spec.add_texture(name="earth", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.12, 0.35, 0.75], rgb2=[0.85, 0.88, 0.92], width=64, height=64,
                     mark=mujoco.mjtMark.mjMARK_RANDOM, markrgb=[0.2, 0.5, 0.25], random=0.3)
    spec.add_material(name="regolith", textures=["", "regolith"], texrepeat=[60, 60], specular=0.05, shininess=0.0, reflectance=0)
    spec.add_material(name="earth", textures=["", "earth"], emission=0.6, specular=0)
    spec.add_material(name="panel", rgba=[0.80, 0.74, 0.62, 1], specular=0.15, shininess=0.1)
    spec.add_material(name="panel_built", rgba=[0.72, 0.68, 0.60, 1], specular=0.1)
    spec.add_material(name="gold_foil", rgba=[0.83, 0.62, 0.20, 1], specular=0.9, shininess=0.9, reflectance=0.15)
    spec.add_material(name="rover", rgba=[0.88, 0.89, 0.91, 1], specular=0.5, shininess=0.5)
    spec.add_material(name="dark_metal", rgba=[0.12, 0.12, 0.14, 1], specular=0.6, shininess=0.6)
    spec.add_material(name="foundation", rgba=[0.50, 0.48, 0.45, 1], specular=0.05)
    spec.add_material(name="target", rgba=[0.15, 0.85, 1.0, 0.25], emission=0.8)
    spec.add_material(name="beacon", rgba=[1.0, 0.25, 0.1, 1], emission=1.0)
    spec.add_material(name="franka_white", rgba=[0.93, 0.93, 0.93, 1], specular=0.5, shininess=0.6)
    spec.add_material(name="franka_dark", rgba=[0.15, 0.15, 0.17, 1], specular=0.5, shininess=0.6)


def build(cfg: SceneConfig = SceneConfig()) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.modelname = "lunar_habitat"
    spec.option.gravity = [0, 0, -cfg.gravity]
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    spec.compiler.degree = False
    spec.visual.headlight.ambient = [0.08, 0.08, 0.09]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 8192
    spec.visual.map.zfar = 200
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.stat.center = [0.6, 0, 0.4]
    spec.stat.extent = 2.0
    _textures(spec)
    wb = spec.worldbody

    # --- lighting: low sun (raking light makes regolith and craters read), faint earthshine fill
    # Spotlight from far away rather than a directional light: its perspective shadow
    # frustum stays tight on the work site, which avoids shadow acne on the heightfield.
    sun_dir = np.array([0.7, 0.45, -0.55])
    sun_dir /= np.linalg.norm(sun_dir)
    sun_pos = np.array([1.0, 0.0, 0.0]) - sun_dir * 9.0
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=sun_pos.tolist(), dir=sun_dir.tolist(),
                 cutoff=30, exponent=1, attenuation=[1, 0, 0], castshadow=True,
                 diffuse=[1.3, 1.25, 1.15], specular=[0.4, 0.4, 0.4], ambient=[0.0, 0.0, 0.0])
    wb.add_light(name="earthshine", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[5, 3, 4], dir=[-0.6, -0.3, -0.5], castshadow=False,
                 diffuse=[0.08, 0.10, 0.16], specular=[0, 0, 0])

    # --- terrain
    res = cfg.terrain_res
    h = _terrain(res, cfg.seed)
    spec.add_hfield(name="terrain", size=[20, 20, 1.2, 0.5], nrow=res, ncol=res, userdata=h.flatten().tolist())
    ground_z = -h[res // 2, int(res * 0.53)] * 1.2  # pad level sits at z = 0
    wb.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain", pos=[0, 0, ground_z],
                material="regolith", friction=[0.9, 0.01, 0.001])

    # --- Earth hanging over the horizon
    wb.add_geom(name="earth", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[4.0, 0, 0], pos=[70, 35, 14],
                material="earth", contype=0, conaffinity=0)

    # --- rover deck carrying the arm
    rover = wb.add_body(name="rover", pos=[0, 0, 0])
    rover.add_geom(name="deck", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.42, 0.34, 0.03], pos=[0.05, 0, DECK_Z - 0.03], material="rover")
    rover.add_geom(name="chassis", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.38, 0.30, 0.09], pos=[0.05, 0, DECK_Z - 0.15], material="gold_foil")
    for i, (x, y) in enumerate([(-0.25, -0.38), (-0.25, 0.38), (0.35, -0.38), (0.35, 0.38)]):
        rover.add_geom(name=f"wheel{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.13, 0.05, 0], pos=[x, y, 0.13],
                       quat=[0.7071, 0.7071, 0, 0], material="dark_metal")
    rover.add_geom(name="mast", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.25, 0], pos=[-0.3, 0.25, DECK_Z + 0.25], material="dark_metal")
    rover.add_geom(name="antenna", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.12, 0.12, 0.03], pos=[-0.3, 0.25, DECK_Z + 0.52],
                   quat=[0.92, 0.38, 0, 0], material="rover", contype=0, conaffinity=0)
    rover.add_geom(name="solar", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.22, 0.14, 0.006], pos=[-0.28, -0.18, DECK_Z + 0.06],
                   quat=[0.96, 0, -0.26, 0], rgba=[0.05, 0.08, 0.25, 1], contype=0, conaffinity=0)
    # panel rack: a low tray with side rails on the deck
    rack_x, rack_y = 0.30, -0.22
    rover.add_geom(name="rack_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.10, 0.18, 0.01], pos=[rack_x, rack_y, DECK_Z + 0.01], material="dark_metal")

    # --- the arm (NVIDIA SimReady Franka, converted)
    arm = mujoco.MjSpec.from_file(str(FRANKA_XML))
    for g in arm.geoms:
        if g.name.endswith("_vis"):
            g.material = "franka_dark" if any(k in g.name for k in ("link7", "hand", "finger", "link0")) else "franka_white"
            g.group = 0
    for m in list(arm.materials):
        arm.delete(m)
    # Franka Hand: 70 N continuous grip force. The converted PhysX drive alone gives only a few newtons.
    for act in arm.actuators:
        if "finger" in act.name:
            act.gainprm[0] = 4000
            act.biasprm[1] = -4000
            act.biasprm[2] = -100
            act.forcerange = [-70, 70]
    mount = rover.add_site(name="arm_mount", pos=[0, 0, DECK_Z])
    mount.attach_body(arm.worldbody.first_body(), "", "")

    # --- habitat foundation + built panels
    site = wb.add_body(name="habitat")
    for k in range(N_SLOTS):
        p, a = slot_pose(k)
        quat = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        site.add_geom(name=f"found{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.15, FOUNDATION_Z / 2],
                      pos=[p[0], p[1], FOUNDATION_Z / 2], quat=quat, material="foundation")
        if k == cfg.gap_slot:
            # guide lips either side of the gap make it a socket, not a flat spot
            for side in (-1, 1):
                off = side * (PANEL[1] + 0.018)
                gx = p[0] - math.sin(a) * off
                gy = p[1] + math.cos(a) * off
                site.add_geom(name=f"lip{side}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.008, 0.02],
                              pos=[gx, gy, FOUNDATION_Z + 0.02], quat=quat, material="dark_metal")
                # front/back rails: a U-channel (7 mm clearance) that holds the seated panel upright
                roff = side * (PANEL[0] + 0.007 + 0.006)
                site.add_geom(name=f"rail{side}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, PANEL[1] + 0.01, 0.0175],
                              pos=[p[0] + math.cos(a) * roff, p[1] + math.sin(a) * roff, FOUNDATION_Z + 0.0175],
                              quat=quat, material="dark_metal")
            continue
        # leave a few open slots far from the robot so the ring reads as "under construction"
        if k in (5, 6, 18, 19, 20):
            continue
        site.add_geom(name=f"built{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=PANEL.tolist(),
                      pos=p.tolist(), quat=quat, material="panel_built")
        if k % 3 == 0:
            site.add_geom(name=f"window{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, PANEL[0] + 0.002, 0],
                          pos=p.tolist(), quat=_mul([math.cos(a / 2), 0, 0, math.sin(a / 2)], [0.7071, 0, 0.7071, 0]),
                          rgba=[0.2, 0.6, 0.9, 1], contype=0, conaffinity=0)

    # the gap's target volume (visual only)
    tp, ta = slot_pose(cfg.gap_slot)
    wb.add_site(name="target", type=mujoco.mjtGeom.mjGEOM_BOX, size=(PANEL * 1.02).tolist(), pos=tp.tolist(),
                quat=[math.cos(ta / 2), 0, 0, math.sin(ta / 2)], material="target", group=1)

    # --- the panel to place (free body)
    panel = wb.add_body(name="panel", pos=[rack_x, rack_y, DECK_Z + 0.02 + PANEL[2]])
    panel.add_freejoint(name="panel_free")
    panel.add_geom(name="panel", type=mujoco.mjtGeom.mjGEOM_BOX, size=PANEL.tolist(), mass=PANEL_MASS, material="panel",
                   friction=[1.5, 0.05, 0.002], solref=[0.004, 1], condim=6)
    # dark grip strip along the top edge (visual only)
    panel.add_geom(name="panel_strip", type=mujoco.mjtGeom.mjGEOM_BOX, size=[PANEL[0] + 0.001, PANEL[1] * 0.4, 0.012],
                   pos=[0, 0, PANEL[2] - 0.012], material="dark_metal", contype=0, conaffinity=0, mass=0)
    panel.add_site(name="panel_center", pos=[0, 0, 0], size=[0.005, 0, 0])

    # --- visual-only flourishes
    if cfg.cinematic:
        rng = np.random.default_rng(cfg.seed + 1)
        for i in range(10):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(2.6, 3.4)
            wb.add_geom(name=f"rock{i}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        size=rng.uniform([0.05, 0.04, 0.03], [0.22, 0.16, 0.10]).tolist(),
                        pos=[RING_CENTER[0] + r * math.cos(a), r * math.sin(a), 0.02],
                        rgba=[0.33, 0.32, 0.31, 1], contype=0, conaffinity=0)
        for i, (x, y) in enumerate([(RING_CENTER[0] + 1.6, 1.6), (RING_CENTER[0] + 1.6, -1.6), (-1.2, 1.4)]):
            wb.add_geom(name=f"pole{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.4, 0], pos=[x, y, 0.4],
                        material="dark_metal", contype=0, conaffinity=0)
            wb.add_geom(name=f"beacon{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], pos=[x, y, 0.82],
                        material="beacon", contype=0, conaffinity=0)

    # cameras
    wb.add_camera(name="hero", pos=[-1.1, -1.9, 1.15], xyaxes=[0.86, -0.51, 0, 0.22, 0.37, 0.90], fovy=50)
    wb.add_camera(name="wide", pos=[-2.6, -3.4, 2.4], xyaxes=[0.8, -0.6, 0, 0.3, 0.4, 0.87], fovy=50)
    wb.add_camera(name="top", pos=[0.7, 0, 3.2], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)
    return spec.compile()


def _mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
