"""Convert an NVIDIA SimReady robot (OpenUSD + UsdPhysics) into a MuJoCo MJCF model.

The USD stage is authored at an arbitrary pose. The kinematic tree at the zero
joint configuration is rebuilt from the joint frames alone: at q = 0 a joint's
frame on body0 (localPos0/localRot0) and on body1 (localPos1/localRot1)
coincide, so  T_parent_child = T(p0, r0) * T(p1, r1)^-1.

Geometry is exported per rigid body, baked into that body's frame, so it does
not depend on the authored pose either.

Usage:
    python tools/usd_to_mjcf.py assets/nvidia/<pkg>/<root>.usda --out assets/converted/<name> \
        --variant MuJoCo=Enabled
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


# ----------------------------------------------------------------------------- math helpers

def quat_to_mat(q: Gf.Quatf | Gf.Quatd) -> np.ndarray:
    w = q.GetReal()
    x, y, z = q.GetImaginary()
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def tf(pos, rot_q) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(rot_q)
    t[:3, 3] = np.array(pos, dtype=float)
    return t


def gf_to_np(m: Gf.Matrix4d) -> np.ndarray:
    # USD matrices are row-vector convention: transpose to column-vector.
    return np.array(m, dtype=float).T


def fmt(v, nd=6) -> str:
    return " ".join(f"{float(x):.{nd}g}" if abs(float(x)) > 1e-12 else "0" for x in v)


# ----------------------------------------------------------------------------- model description

@dataclass
class Joint:
    name: str
    kind: str  # hinge | slide
    parent: str
    child: str
    t_parent_joint: np.ndarray
    t_child_joint: np.ndarray
    axis_local: np.ndarray  # axis in joint frame
    limits: tuple[float, float] | None
    stiffness: float | None = None
    damping: float | None = None
    max_force: float | None = None


@dataclass
class Body:
    path: str
    name: str
    mass: float | None = None
    com: np.ndarray | None = None
    diag_inertia: np.ndarray | None = None
    principal_axes: np.ndarray | None = None  # quat wxyz
    meshes: list[tuple[str, trimesh.Trimesh, tuple[float, float, float], bool]] = field(default_factory=list)


def bound_color(prim: Usd.Prim) -> tuple[float, float, float]:
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if mat:
        for shader_prim in Usd.PrimRange(mat.GetPrim()):
            sh = UsdShade.Shader(shader_prim)
            if not sh:
                continue
            for name in ("diffuseColor", "diffuse_color_constant", "base_color"):
                inp = sh.GetInput(name)
                if inp and inp.Get() is not None:
                    c = inp.Get()
                    try:
                        return float(c[0]), float(c[1]), float(c[2])
                    except TypeError:
                        pass
    dc = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get() if UsdGeom.Gprim(prim) else None
    if dc:
        return tuple(float(x) for x in dc[0])
    return (0.8, 0.8, 0.8)


def mesh_in_frame(prim: Usd.Prim, frame_world: np.ndarray, xc: UsdGeom.XformCache) -> trimesh.Trimesh | None:
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts:
        return None
    pts = np.array(pts, dtype=float)
    idx = np.array(idx, dtype=np.int64)
    faces, k = [], 0
    for c in counts:
        for i in range(1, c - 1):  # fan triangulation
            faces.append((idx[k], idx[k + i], idx[k + i + 1]))
        k += c
    world = gf_to_np(xc.GetLocalToWorldTransform(prim))
    rel = np.linalg.inv(frame_world) @ world
    v = (rel[:3, :3] @ pts.T).T + rel[:3, 3]
    m = trimesh.Trimesh(vertices=v, faces=np.array(faces), process=True)
    return m


class Converter:
    def __init__(self, usd_path: Path, variants: dict[str, str], meters_scale: float | None):
        self.stage = Usd.Stage.Open(str(usd_path))
        root = self.stage.GetDefaultPrim()
        vsets = root.GetVariantSets()
        for k, v in variants.items():
            vsets.GetVariantSet(k).SetVariantSelection(v)
        self.root = root
        self.mpu = meters_scale or UsdGeom.GetStageMetersPerUnit(self.stage)
        self.xc = UsdGeom.XformCache()
        self.bodies: dict[str, Body] = {}
        self.joints: list[Joint] = []
        self.fixed_root: str | None = None

    @staticmethod
    def _name(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    def collect(self) -> None:
        for p in Usd.PrimRange(self.root):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                b = Body(str(p.GetPath()), self._name(str(p.GetPath())))
                m = UsdPhysics.MassAPI(p)
                if p.HasAPI(UsdPhysics.MassAPI):
                    mass = m.GetMassAttr().Get()
                    b.mass = float(mass) if mass and mass > 0 else None
                    com = m.GetCenterOfMassAttr().Get()
                    if com is not None and not any(math.isinf(c) or abs(c) > 1e6 for c in com):
                        b.com = np.array(com, dtype=float)
                    di = m.GetDiagonalInertiaAttr().Get()
                    if di is not None and all(x > 0 for x in di):
                        b.diag_inertia = np.array(di, dtype=float)
                        pa = m.GetPrincipalAxesAttr().Get()
                        if pa is not None and pa.GetReal() != 0 or (pa is not None and any(pa.GetImaginary())):
                            b.principal_axes = mat_to_quat(quat_to_mat(pa))
                self.bodies[b.path] = b
        for p in Usd.PrimRange(self.root):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not b1:
                continue
            if not b0:
                self.fixed_root = b1[0]
                continue
            t0 = tf(j.GetLocalPos0Attr().Get() or (0, 0, 0), j.GetLocalRot0Attr().Get() or Gf.Quatf(1))
            t1 = tf(j.GetLocalPos1Attr().Get() or (0, 0, 0), j.GetLocalRot1Attr().Get() or Gf.Quatf(1))
            kind, axis, limits = "fixed", np.zeros(3), None
            if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
                rev = p.IsA(UsdPhysics.RevoluteJoint)
                jj = UsdPhysics.RevoluteJoint(p) if rev else UsdPhysics.PrismaticJoint(p)
                kind = "hinge" if rev else "slide"
                axis = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[jj.GetAxisAttr().Get()]
                lo, hi = jj.GetLowerLimitAttr().Get(), jj.GetUpperLimitAttr().Get()
                if lo is not None and hi is not None and abs(lo) < 1e6 and abs(hi) < 1e6:
                    limits = (math.radians(lo), math.radians(hi)) if rev else (lo, hi)
            joint = Joint(self._name(str(p.GetPath())), kind, b0[0], b1[0], t0, t1, axis, limits)
            drive_type = "angular" if kind == "hinge" else "linear"
            if p.HasAPI(UsdPhysics.DriveAPI, drive_type):
                d = UsdPhysics.DriveAPI(p, drive_type)
                joint.stiffness = d.GetStiffnessAttr().Get()
                joint.damping = d.GetDampingAttr().Get()
                joint.max_force = d.GetMaxForceAttr().Get()
            self.joints.append(joint)

    def collect_meshes(self) -> None:
        body_paths = sorted(self.bodies, key=len, reverse=True)
        for p in Usd.PrimRange(self.root):
            if p.GetTypeName() != "Mesh":
                continue
            if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            path = str(p.GetPath())
            owner = next((b for b in body_paths if path.startswith(b + "/")), None)
            if owner is None:
                continue
            frame = gf_to_np(self.xc.GetLocalToWorldTransform(self.stage.GetPrimAtPath(owner)))
            m = mesh_in_frame(p, frame, self.xc)
            if m is None or len(m.faces) == 0:
                continue
            collide = p.HasAPI(UsdPhysics.CollisionAPI)
            self.bodies[owner].meshes.append((self._name(path), m, bound_color(p), collide))

    # ------------------------------------------------------------------------- MJCF writing
    def write(self, out_dir: Path, model_name: str) -> Path:
        mesh_dir = out_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        s = self.mpu
        children: dict[str, list[Joint]] = {}
        for j in self.joints:
            children.setdefault(j.parent, []).append(j)
        child_set = {j.child for j in self.joints}
        roots = [b for b in self.bodies if b not in child_set]
        root_body = self.fixed_root if self.fixed_root in roots else roots[0]

        mj = ET.Element("mujoco", model=model_name)
        ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes", autolimits="true")
        ET.SubElement(mj, "default")
        asset = ET.SubElement(mj, "asset")
        world = ET.SubElement(mj, "worldbody")
        actuator = ET.SubElement(mj, "actuator")
        equality = ET.SubElement(mj, "equality")

        materials: dict[tuple, str] = {}

        def material_for(rgb) -> str:
            key = tuple(round(c, 3) for c in rgb)
            if key not in materials:
                name = f"mat{len(materials)}"
                materials[key] = name
                ET.SubElement(asset, "material", name=name, rgba=fmt([*key, 1.0], 4), specular="0.4", shininess="0.4")
            return materials[key]

        def emit_body(parent_el, body_path: str, t_parent_body: np.ndarray, joint: Joint | None):
            b = self.bodies[body_path]
            el = ET.SubElement(
                parent_el, "body", name=b.name,
                pos=fmt(t_parent_body[:3, 3] * s), quat=fmt(mat_to_quat(t_parent_body[:3, :3])),
            )
            if b.mass:
                attrs = {"mass": f"{b.mass:.6g}", "pos": fmt((b.com if b.com is not None else np.zeros(3)) * s)}
                if b.diag_inertia is not None:
                    attrs["diaginertia"] = fmt(b.diag_inertia * s * s)
                    if b.principal_axes is not None:
                        attrs["quat"] = fmt(b.principal_axes)
                    ET.SubElement(el, "inertial", **attrs)
            if joint is not None and joint.kind != "fixed":
                axis_child = joint.t_child_joint[:3, :3] @ joint.axis_local
                jattrs = dict(name=joint.name, type=joint.kind, pos=fmt(joint.t_child_joint[:3, 3] * s), axis=fmt(axis_child))
                if joint.limits:
                    lo, hi = joint.limits
                    if joint.kind == "slide":
                        lo, hi = lo * s, hi * s
                    jattrs["range"] = fmt([lo, hi])
                jattrs["damping"] = "0.1" if joint.kind == "hinge" else "1"
                jattrs["armature"] = "0.1" if joint.kind == "hinge" else "0.01"
                ET.SubElement(el, "joint", **jattrs)
            for i, (mname, mesh, rgb, collide) in enumerate(b.meshes):
                fname = f"{b.name}_{i}.obj"
                m = mesh.copy()
                m.apply_scale(s)
                m.export(mesh_dir / fname)
                aname = f"{b.name}_{i}"
                ET.SubElement(asset, "mesh", name=aname, file=fname)
                ET.SubElement(el, "geom", name=f"{aname}_vis", type="mesh", mesh=aname, material=material_for(rgb),
                              contype="0", conaffinity="0", group="2", density="0")
                if collide:
                    ET.SubElement(el, "geom", name=f"{aname}_col", type="mesh", mesh=aname, group="3",
                                  rgba="0.5 0.5 0.5 0.3", **({"density": "0"} if b.mass else {}))
            for cj in children.get(body_path, []):
                t_pc = cj.t_parent_joint @ np.linalg.inv(cj.t_child_joint)
                emit_body(el, cj.child, t_pc, cj)

        emit_body(world, root_body, np.eye(4), None)

        # Actuators from drives; joints without a drive that share a parent+kind with a driven
        # gripper joint are coupled with an equality (USD mimic joints).
        driven = [j for j in self.joints if j.stiffness]
        for j in driven:
            kp = j.stiffness * (180 / math.pi if j.kind == "hinge" else 1.0)  # PhysX angular drives are per degree
            kv = (j.damping or 0) * (180 / math.pi if j.kind == "hinge" else 1.0)
            attrs = dict(name=f"act_{j.name}", joint=j.name, kp=f"{kp:.6g}", kv=f"{kv:.6g}")
            if j.limits:
                lo, hi = j.limits
                if j.kind == "slide":
                    lo, hi = lo * s, hi * s
                attrs["ctrlrange"] = fmt([lo, hi])
            if j.max_force and j.max_force < 1e6:
                attrs["forcerange"] = fmt([-j.max_force, j.max_force])
            ET.SubElement(actuator, "position", **attrs)
        driven_names = {j.name for j in driven}
        for j in self.joints:
            if j.kind == "fixed" or j.name in driven_names:
                continue
            mate = next((d for d in driven if d.parent == j.parent and d.kind == j.kind), None)
            if mate:
                ET.SubElement(equality, "joint", joint1=j.name, joint2=mate.name, polycoef="0 1 0 0 0",
                              solref="0.005 1", solimp="0.95 0.99 0.001")

        ET.indent(mj)
        path = out_dir / f"{model_name}.xml"
        ET.ElementTree(mj).write(path)
        return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("usd", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--variant", action="append", default=[], help="VariantSet=Selection on the default prim")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    args = ap.parse_args()
    variants = dict(v.split("=", 1) for v in args.variant)
    c = Converter(args.usd, variants, args.meters_per_unit)
    c.collect()
    c.collect_meshes()
    name = args.name or args.out.name
    path = c.write(args.out, name)
    print(f"bodies={len(c.bodies)} joints={len(c.joints)} meshes={sum(len(b.meshes) for b in c.bodies.values())} -> {path}")


if __name__ == "__main__":
    main()
