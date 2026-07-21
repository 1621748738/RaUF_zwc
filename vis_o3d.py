"""Standalone open3d visualizer for RaUF inference results.

Copy this file and the *_pred.ply / *_gt.ply files to your Windows machine,
then run:

    python vis_o3d.py output_infer/02161              # single sample
    python vis_o3d.py output_infer/ --slide-show      # cycle through all samples
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def load_ply_points(path: str) -> np.ndarray:
    """Read XYZ from an ASCII PLY file.  Returns (N, 3) float32 array."""
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)


def make_pcd(points: np.ndarray, color: tuple) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def vis_single(pred_path: str, gt_path: str, window_name: str = "RaUF"):
    pred_pts = load_ply_points(pred_path)
    gt_pts = load_ply_points(gt_path)
    print(f"Pred: {len(pred_pts)} points, GT: {len(gt_pts)} points")

    pred = make_pcd(pred_pts, (0.2, 0.8, 0.2))   # green
    gt = make_pcd(gt_pts, (1.0, 0.2, 0.2))         # red

    # Coordinate frame for reference
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    o3d.visualization.draw_geometries(
        [gt, pred, frame],
        window_name=f"{window_name}  |  red=GT  green=Pred",
        width=1600, height=900,
    )


def vis_slide_show(pred_paths: list, gt_paths: list):
    """Cycle through samples with keyboard.  Press N for next, Q to quit."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("RaUF  |  red=GT  green=Pred  |  N=next  Q=quit",
                      width=1600, height=900)

    idx = [0]
    pred_all = [(load_ply_points(p), load_ply_points(g)) for p, g in zip(pred_paths, gt_paths)]

    def load_sample(i):
        vis.clear_geometries()
        pts_p, pts_g = pred_all[i]
        vis.add_geometry(make_pcd(pts_p, (0.2, 0.8, 0.2)))
        vis.add_geometry(make_pcd(pts_g, (1.0, 0.2, 0.2)))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(frame)
        vis.reset_view_point(True)
        print(f"[{i + 1}/{len(pred_paths)}]  Pred: {len(pts_p)}  GT: {len(pts_g)}")

    def next_sample(_vis):
        idx[0] = (idx[0] + 1) % len(pred_paths)
        load_sample(idx[0])

    def prev_sample(_vis):
        idx[0] = (idx[0] - 1) % len(pred_paths)
        load_sample(idx[0])

    vis.register_key_callback(ord("N"), next_sample)
    vis.register_key_callback(ord("P"), prev_sample)
    vis.register_key_callback(ord("Q"), lambda v: vis.close())

    load_sample(0)
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser("open3d RaUF point cloud visualizer")
    parser.add_argument("input", help="sample prefix (e.g. output_infer/02161) or directory")
    parser.add_argument("--slide-show", action="store_true",
                        help="cycle through all samples with N/P keys")
    args = parser.parse_args()

    base = Path(args.input)

    if args.slide_show or base.is_dir():
        base = base if base.is_dir() else base.parent
        pred_files = sorted(base.glob("*_pred.ply"))
        gt_files = sorted(base.glob("*_gt.ply"))
        if not pred_files:
            print(f"No *_pred.ply files found in {base}")
            sys.exit(1)
        vis_slide_show(pred_files, gt_files)
    else:
        dirname = base.parent
        stem = base.name  # e.g. "02161" or "02161_pred" or "02161_pred.ply"

        # Strip _pred and _gt suffixes + .ply extension to get the base sample ID
        for suffix in ("_pred.ply", "_gt.ply", "_pred", "_gt", ".ply"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        pred_p = str(dirname / f"{stem}_pred.ply")
        gt_p = str(dirname / f"{stem}_gt.ply")
        vis_single(pred_p, gt_p, window_name=stem)


if __name__ == "__main__":
    main()

# 路径
# cd C:\Users\16217\Desktop\毕业论文下一章节\环境感知\RaLD-RaUF-final-ColoRadar-RaDelft-VoD
# python vis_o3d.py output_infer/02161          # 只传样本前缀即可
# python vis_o3d.py output_infer/full/00009 
# python vis_o3d.py output_infer/02161_pred.ply # 传完整文件名也可以
# 文件在output_infer