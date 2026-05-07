#!/usr/bin/env python3

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVAL_ROOT = Path.home() / "robot_description" / "sfm_dataset" / "eval_results"
CSV_PATH = EVAL_ROOT / "experiment_log.csv"


def to_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def sample_points(points, max_points=40000):
    points = np.asarray(points)

    if len(points) <= max_points:
        return points

    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def load_cloud(path):
    if path is None:
        return None

    path = Path(os.path.expanduser(str(path)))

    if not path.exists():
        return None

    pcd = o3d.io.read_point_cloud(str(path))

    if pcd is None or len(pcd.points) == 0:
        return None

    return pcd


def get_estimated_pcd_path(trial_dir):
    candidates = [
        trial_dir / "pcd" / "auto_grasp_clean_object.ply",
        trial_dir / "pcd" / "05_after_dbscan_selected_clusters.ply",
        trial_dir / "pcd" / "04_after_plane_removal.ply",
        trial_dir / "pcd" / "04_plane_removal_skipped.ply",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def plot_trial(row, trial_index):
    trial_id = row.get("trial_id", "")
    trial_dir = EVAL_ROOT / trial_id

    if not trial_dir.exists():
        print(f"[WARN] Missing trial directory: {trial_dir}")
        return False

    fig_dir = trial_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    est_pcd_path = get_estimated_pcd_path(trial_dir)
    est_pcd = load_cloud(est_pcd_path)

    if est_pcd is None:
        print(f"[WARN] Missing estimated PCD for {trial_id}")
        return False

    est_points = sample_points(np.asarray(est_pcd.points), max_points=40000)

    gt_pcd_path = row.get("gt_pcd_path", "")
    gt_pcd = load_cloud(gt_pcd_path)

    gt_points = None
    if gt_pcd is not None:
        gt_points = sample_points(np.asarray(gt_pcd.points), max_points=40000)
    else:
        print(f"[WARN] GT cloud missing or empty for {trial_id}: {gt_pcd_path}")

    grasp_x = to_float(row, "grasp_x")
    grasp_y = to_float(row, "grasp_y")
    grasp_yaw_deg = to_float(row, "grasp_yaw_deg")

    gt_x = to_float(row, "gt_center_x")
    gt_y = to_float(row, "gt_center_y")

    depth_model = row.get("depth_model", "unknown")
    object_name = row.get("gt_object_name", "unknown object")

    fig_path = fig_dir / "topdown_grasp_result.png"
    backup_path = fig_dir / "topdown_grasp_result_old.png"
    extra_path = fig_dir / "topdown_grasp_result_with_gt.png"

    if fig_path.exists() and not backup_path.exists():
        shutil.copy2(fig_path, backup_path)

    plt.figure(figsize=(7.0, 6.4))
    ax = plt.gca()

    gt_handle = None
    if gt_points is not None and len(gt_points) > 0:
        gt_handle = ax.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            s=2.0,
            alpha=0.28,
            c="green",
            label="Ground-truth object cloud",
            zorder=1,
        )

    est_handle = ax.scatter(
        est_points[:, 0],
        est_points[:, 1],
        s=7.0,
        alpha=0.75,
        c="tab:blue",
        label="Estimated object cloud",
        zorder=3,
    )

    grasp_handle = None
    if np.isfinite(grasp_x) and np.isfinite(grasp_y):
        grasp_handle = ax.scatter(
            [grasp_x],
            [grasp_y],
            marker="x",
            s=140,
            c="red",
            linewidths=2.2,
            label="Estimated grasp position",
            zorder=5,
        )

        if np.isfinite(grasp_yaw_deg):
            yaw = np.deg2rad(grasp_yaw_deg)
            arrow_length = 0.08

            ax.arrow(
                grasp_x,
                grasp_y,
                arrow_length * np.cos(yaw),
                arrow_length * np.sin(yaw),
                head_width=0.010,
                head_length=0.014,
                length_includes_head=True,
                color="red",
                linewidth=2.2,
                zorder=5,
            )

    gt_centre_handle = None
    if np.isfinite(gt_x) and np.isfinite(gt_y):
        gt_centre_handle = ax.scatter(
            [gt_x],
            [gt_y],
            marker="o",
            s=150,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="Ground-truth object centre",
            zorder=6,
        )

    all_xy = [est_points[:, :2]]

    if gt_points is not None and len(gt_points) > 0:
        all_xy.append(gt_points[:, :2])

    extra_xy = []
    if np.isfinite(grasp_x) and np.isfinite(grasp_y):
        extra_xy.append([grasp_x, grasp_y])
    if np.isfinite(gt_x) and np.isfinite(gt_y):
        extra_xy.append([gt_x, gt_y])

    if len(extra_xy) > 0:
        all_xy.append(np.asarray(extra_xy))

    all_xy = np.vstack(all_xy)

    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)

    x_margin = max(0.015, (x_max - x_min) * 0.12)
    y_margin = max(0.015, (y_max - y_min) * 0.12)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_xlabel("x in base_link (m)")
    ax.set_ylabel("y in base_link (m)")
    ax.set_title(
        f"Top-down object cloud and estimated grasp pose\n"
        f"{object_name} | {depth_model}"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.4)

    handles = []
    labels = []

    if est_handle is not None:
        handles.append(est_handle)
        labels.append("Estimated object cloud")

    if gt_handle is not None:
        handles.append(gt_handle)
        labels.append("Ground-truth object cloud")

    if grasp_handle is not None:
        handles.append(grasp_handle)
        labels.append("Estimated grasp position")

    if gt_centre_handle is not None:
        handles.append(gt_centre_handle)
        labels.append("Ground-truth object centre")

    ax.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=250)
    plt.savefig(extra_path, dpi=250)
    plt.close()

    print(f"[OK] Regenerated {trial_index:02d}: {fig_path}")
    return True


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find {CSV_PATH}")

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        raise RuntimeError("No rows in experiment_log.csv")

    ok_count = 0

    for i, row in enumerate(rows, start=1):
        if plot_trial(row, i):
            ok_count += 1

    print()
    print(f"Done. Regenerated {ok_count}/{len(rows)} top-down figures.")
    print(f"Eval root: {EVAL_ROOT}")


if __name__ == "__main__":
    main()
