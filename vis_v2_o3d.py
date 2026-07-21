"""Standalone open3d visualizer for RaUF V2 inference output.

Copy to Windows along with the output_v2/ directory, then:

    pip install open3d numpy

    python vis_v2_o3d.py output_v2_smoke/02161          # single frame
    python vis_v2_o3d.py output_v2_smoke/ --slide-show   # cycle through frames (N/P keys)

Displays:
    - Red spheres:    component centres (size = probability)
    - Blue wireframe: 95% confidence ellipsoids (only if *_ellipsoid_components.ply exists)
    - Green points:   probability point cloud
    - Cyan wireframe: coordinate frame

Keyboard (slide-show mode):  N=next  P=prev  Q=quit
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ═══════════════════════════════════════════════════════════════════════════════
# Binary PLY reader  (handles standard header + raw float32 data)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_ply_header(path: str) -> tuple[int, list[str], int]:
    """Return (vertex_count, property_names, header_bytes)."""
    with open(path, "rb") as f:
        header = b""
        count = 0
        props = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Corrupt PLY: {path}")
            header += line
            if line.strip() == b"end_header":
                break
            parts = line.decode("ascii").strip().split()
            if parts[0] == "element" and parts[1] == "vertex":
                count = int(parts[2])
            elif parts[0] == "property" and parts[1] == "float":
                props.append(parts[2])
        return count, props, len(header)


def _read_binary_ply(path: str) -> dict[str, np.ndarray]:
    """Read a binary little-endian PLY. Returns {prop_name: (N,) array}."""
    n, props, hdr_bytes = _parse_ply_header(path)
    dtype = np.dtype([(p, "<f4") for p in props])
    with open(path, "rb") as f:
        f.seek(hdr_bytes)
        data = np.frombuffer(f.read(), dtype=dtype, count=n)
    result = {}
    for p in props:
        result[p] = data[p].astype(np.float32)
    # Add convenience: xyz as (N,3)
    if {"x", "y", "z"} <= set(props):
        result["xyz"] = np.column_stack([data["x"], data["y"], data["z"]])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Ellipsoid mesh builder
# ═══════════════════════════════════════════════════════════════════════════════

def _make_unit_ellipsoid_mesh(n_rings: int = 12, n_segments: int = 16):
    """Triangle mesh of a unit sphere, subdivided."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=n_rings)
    sphere.compute_vertex_normals()
    return sphere


_UNIT_SPHERE_MESH = None


def build_ellipsoid_mesh(center: np.ndarray,
                         semi_axes: np.ndarray,   # [3]  in descending order
                         rotation: np.ndarray,     # [3,3] column-major (eigenvectors)
                         color: tuple = (0.2, 0.4, 1.0),
                         opacity: float = 0.3) -> o3d.geometry.TriangleMesh:
    """Create an ellipsoid wireframe mesh at *center* with given axes and *rotation*."""
    global _UNIT_SPHERE_MESH
    if _UNIT_SPHERE_MESH is None:
        _UNIT_SPHERE_MESH = _make_unit_ellipsoid_mesh(12, 16)

    ell = o3d.geometry.TriangleMesh(_UNIT_SPHERE_MESH)
    # Scale: apply diag(semi_axes) → vertex-wise multiply
    verts = np.asarray(ell.vertices) * semi_axes[None, :]  # [Nv, 3]
    # Rotate: R @ v^T  for each vertex (assuming R columns are principal axes)
    verts = verts @ rotation.T
    # Translate
    verts = verts + center[None, :]
    ell.vertices = o3d.utility.Vector3dVector(verts)
    ell.compute_vertex_normals()
    ell.paint_uniform_color(color)
    return ell


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def vis_single(sid: str, out_dir: Path):
    """Display a single frame with all available V2 outputs."""
    comp_path = out_dir / f"{sid}_components.ply"
    ell_comp_path = out_dir / f"{sid}_ellipsoid_components.ply"
    prob_path = out_dir / f"{sid}_probability.ply"

    geoms = []

    # Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geoms.append(frame)

    # ---- Components as spheres (top 2000 by probability for performance) ----
    if comp_path.exists():
        comp = _read_binary_ply(str(comp_path))
        xyz = comp["xyz"]
        prob = comp.get("probability", np.ones(len(xyz)))
        # Sort by probability, take top 2000
        if len(xyz) > 2000:
            idx = np.argsort(prob)[-2000:]
            xyz, prob = xyz[idx], prob[idx]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # Color by probability (blue = low, red = high)
        colors = np.zeros((len(xyz), 3), dtype=np.float64)
        p_norm = np.clip(prob / (prob.max() + 1e-10), 0, 1)
        colors[:, 0] = p_norm   # red channel
        colors[:, 2] = 1.0 - p_norm  # blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(pcd)
        print(f"  Components: {len(xyz)} points (top 2000 shown)")

    # ---- Ellipsoid wireframes ----
    if ell_comp_path.exists():
        ell = _read_binary_ply(str(ell_comp_path))
        n = len(ell["xyz"])
        # Show at most 200 ellipsoids
        step = max(1, n // 200)
        for i in range(0, n, step):
            axes = np.array([ell[f"semi_axis_{j}"][i] for j in (1, 2, 3)])
            rot = np.array([[ell[f"rot_{r}{c}"][i] for c in (1, 2, 3)] for r in (1, 2, 3)])
            center = ell["xyz"][i]
            mesh = build_ellipsoid_mesh(center, axes, rot,
                                        color=(0.2, 0.4, 1.0), opacity=0.3)
            # Wireframe
            wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            wire.paint_uniform_color((0.2, 0.5, 1.0))
            geoms.append(wire)
        print(f"  Ellipsoids: {n} components, showing {n // step} wireframes")

    # ---- Probability point cloud ----
    if prob_path.exists():
        prob = _read_binary_ply(str(prob_path))
        xyz = prob["xyz"]
        p = prob.get("probability", np.ones(len(xyz)))
        pcd_p = o3d.geometry.PointCloud()
        pcd_p.points = o3d.utility.Vector3dVector(xyz)
        # Green intensity by probability
        colors = np.zeros((len(xyz), 3), dtype=np.float64)
        colors[:, 1] = np.clip(p / (p.max() + 1e-10), 0.3, 1.0)
        pcd_p.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(pcd_p)
        print(f"  Probability: {len(xyz)} voxels")

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"RaUF V2  |  {sid}  |  blue=ellipsoids  red=components  green=probability",
        width=1600, height=900,
    )


def vis_slide_show(out_dir: Path):
    """Cycle through all samples with N/P keys."""
    samples = sorted(set(
        p.stem.replace("_components", "").replace("_ellipsoid_components", "")
        .replace("_probability", "").replace("_ellipsoids", "")
        for p in out_dir.glob("*.ply")
    ))
    if not samples:
        print(f"No PLY files found in {out_dir}")
        sys.exit(1)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("RaUF V2 Slide Show  |  N=next P=prev Q=quit",
                      width=1600, height=900)
    idx = [0]

    def load_sample(i):
        vis.clear_geometries()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(frame, reset_bounding_box=False)

        sid = samples[i]
        comp_path = out_dir / f"{sid}_components.ply"
        ell_comp_path = out_dir / f"{sid}_ellipsoid_components.ply"
        prob_path = out_dir / f"{sid}_probability.ply"

        if comp_path.exists():
            comp = _read_binary_ply(str(comp_path))
            xyz = comp["xyz"]
            prob = comp.get("probability", np.ones(len(xyz)))
            if len(xyz) > 2000:
                ord_idx = np.argsort(prob)[-2000:]
                xyz, prob = xyz[ord_idx], prob[ord_idx]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            colors = np.zeros((len(xyz), 3), dtype=np.float64)
            p_norm = np.clip(prob / (prob.max() + 1e-10), 0, 1)
            colors[:, 0] = p_norm
            colors[:, 2] = 1.0 - p_norm
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pcd, reset_bounding_box=False)

        if ell_comp_path.exists():
            ell = _read_binary_ply(str(ell_comp_path))
            n = len(ell["xyz"])
            step = max(1, n // 200)
            for j in range(0, n, step):
                axes = np.array([ell[f"semi_axis_{k}"][j] for k in (1, 2, 3)])
                rot = np.array([[ell[f"rot_{r}{c}"][j] for c in (1, 2, 3)]
                                for r in (1, 2, 3)])
                mesh = build_ellipsoid_mesh(ell["xyz"][j], axes, rot)
                wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                wire.paint_uniform_color((0.2, 0.5, 1.0))
                vis.add_geometry(wire, reset_bounding_box=False)

        if prob_path.exists():
            prob = _read_binary_ply(str(prob_path))
            pcd_p = o3d.geometry.PointCloud()
            pcd_p.points = o3d.utility.Vector3dVector(prob["xyz"])
            colors = np.zeros((len(prob["xyz"]), 3), dtype=np.float64)
            p_vals = prob.get("probability", np.ones(len(prob["xyz"])))
            colors[:, 1] = np.clip(p_vals / (p_vals.max() + 1e-10), 0.3, 1.0)
            pcd_p.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pcd_p, reset_bounding_box=False)

        vis.reset_view_point(True)
        print(f"[{i + 1}/{len(samples)}] {sid}")

    def next_sample(_vis):
        idx[0] = (idx[0] + 1) % len(samples)
        load_sample(idx[0])

    def prev_sample(_vis):
        idx[0] = (idx[0] - 1) % len(samples)
        load_sample(idx[0])

    vis.register_key_callback(ord("N"), next_sample)
    vis.register_key_callback(ord("P"), prev_sample)
    vis.register_key_callback(ord("Q"), lambda v: vis.close())

    load_sample(0)
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser("open3d V2 RaUF visualizer")
    parser.add_argument("input", help="sample prefix (e.g. output_v2_smoke/02161) or directory")
    parser.add_argument("--slide-show", action="store_true",
                        help="cycle through all samples with N/P keys")
    args = parser.parse_args()

    base = Path(args.input)

    if args.slide_show or base.is_dir():
        vis_slide_show(base if base.is_dir() else base.parent)
    else:
        sid = base.name if not base.name.endswith("_components") else \
            base.name.rsplit("_components", 1)[0]
        vis_single(sid, base.parent)


if __name__ == "__main__":
    main()


# # 单帧
# a

# # 幻灯片模式，N=下一帧 P=上一帧 Q=退出
# python vis_v2_o3d.py output_v2_smoke/ --slide-show
# 显示内容：

# 颜色	含义
# 蓝色线框	95% 置信椭球（top-500 组件）
# 红→蓝渐变	组件中心（红=高概率，蓝=低概率）
# 绿色	概率点云（体素颜色深浅=概率高低）
