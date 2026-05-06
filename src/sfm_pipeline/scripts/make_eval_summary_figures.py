#!/usr/bin/env python3

import csv
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = Path.home() / "robot_description" / "sfm_dataset" / "eval_results" / "experiment_log.csv"
OUT_DIR = Path.home() / "robot_description" / "sfm_dataset" / "eval_results" / "summary_figures"


STAGE_KEYS = [
    "stage_00_global_fused",
    "stage_01_input_downsampled",
    "stage_01b_table_height_filter",
    "stage_02_radius_outlier",
    "stage_03_statistical_outlier",
    "stage_04_object_candidate",
    "stage_05_dbscan_selected",
]

STAGE_LABELS = [
    "Global fused",
    "Downsampled",
    "Height filter",
    "Radius filter",
    "Stat filter",
    "Candidate",
    "DBSCAN",
]

SUMMARY_METRICS = [
    ("center_error_xy_mm", "Centre XY error mm"),
    ("center_error_3d_mm", "Centre 3D error mm"),
    ("top_z_error_mm", "Top height error mm"),
    ("yaw_error_deg", "Yaw error deg"),
    ("cloud_est_to_gt_mean_mm", "Cloud est-to-GT mean distance mm"),
    ("cloud_est_to_gt_median_mm", "Cloud est-to-GT median distance mm"),
    ("cloud_est_to_gt_p95_mm", "Cloud est-to-GT 95th percentile distance mm"),
    ("cloud_gt_to_est_mean_mm", "Cloud GT-to-est mean distance mm"),
    ("cloud_gt_to_est_median_mm", "Cloud GT-to-est median distance mm"),
    ("cloud_gt_to_est_p95_mm", "Cloud GT-to-est 95th percentile distance mm"),
    ("final_object_points", "Final object points"),
    ("total_compute_time_s", "Total compute time s"),
    ("mean_frame_processing_time_s", "Mean frame processing time s"),
]

# Intended experimental profiles.
# If these parameter values are not present in experiment_log.csv, this script
# uses this table so that the summary still shows the intended model-specific setup.
MODEL_CONFIG_PROFILES = {
    "depth_pro": {
        "profile_name": "Depth Pro / plane OFF",
        "enable_plane_removal": "False",
        "plane_distance_threshold": "-",
        "object_above_plane_threshold": "-",
        "plane_min_inlier_ratio": "-",
        "plane_min_normal_z": "-",
        "notes": "Table-depth scaling + height filter + outlier removal + DBSCAN",
    },
    "zoedepth": {
        "profile_name": "ZoeDepth / plane ON",
        "enable_plane_removal": "True",
        "plane_distance_threshold": "0.018",
        "object_above_plane_threshold": "0.010",
        "plane_min_inlier_ratio": "0.05",
        "plane_min_normal_z": "0.55",
        "notes": "Plane removal enabled because tilted planar artefacts were observed",
    },
    "depth_anything_v2_metric": {
        "profile_name": "Depth Anything V2 / plane ON",
        "enable_plane_removal": "True",
        "plane_distance_threshold": "0.015",
        "object_above_plane_threshold": "0.010",
        "plane_min_inlier_ratio": "0.05",
        "plane_min_normal_z": "0.55",
        "notes": "Plane removal enabled because stronger table/background artefacts were observed",
    },
}

COMMON_CONFIG_DEFAULTS = {
    "grasp_model": "pca",
    "use_table_depth_scale": "True",
    "table_z": "0.0",
    "table_scale_v_min_ratio": "0.60",
    "table_scale_v_max_ratio": "0.92",
    "enable_fused_table_z_alignment": "True",
    "table_alignment_percentile": "2.0",
    "max_table_z_correction": "0.080",
    "enable_table_height_filter": "True",
    "min_object_height_above_table": "0.006",
    "max_object_height_above_table": "0.320",
    "enable_depth_percentile_filter": "True",
    "depth_percentile_min": "0.5",
    "depth_percentile_max": "99.5",
    "pixel_step": "3",
    "frame_voxel_size": "0.006",
    "final_voxel_size": "0.005",
    "enable_radius_outlier_removal": "True",
    "radius_outlier_nb_points": "5",
    "radius_outlier_radius": "0.035",
    "enable_statistical_outlier_removal": "True",
    "outlier_nb_neighbors": "20",
    "outlier_std_ratio": "1.8",
    "dbscan_eps": "0.045",
    "dbscan_min_points": "12",
    "keep_nearby_clusters": "False",
}

PER_TRIAL_CONFIG_COLUMNS = [
    ("profile_name", "Configuration profile"),
    ("use_table_depth_scale", "Table-depth scale"),
    ("table_z", "Table z"),
    ("enable_fused_table_z_alignment", "Fused z alignment"),
    ("enable_table_height_filter", "Height filter"),
    ("enable_plane_removal", "Plane removal"),
    ("plane_distance_threshold", "Plane dist."),
    ("object_above_plane_threshold", "Above-plane th."),
    ("plane_min_normal_z", "Plane min normal-z"),
    ("pixel_step", "Pixel step"),
    ("frame_voxel_size", "Frame voxel"),
    ("final_voxel_size", "Final voxel"),
    ("dbscan_eps", "DBSCAN eps"),
    ("dbscan_min_points", "DBSCAN min pts"),
]


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return np.nan


def valid_values(rows, key):
    values = np.array([to_float(r, key) for r in rows], dtype=float)
    return values[np.isfinite(values)]


def row_has_value(row, key):
    if key not in row:
        return False

    value = row.get(key, "")

    if value is None:
        return False

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return False

    return True


def normalise_depth_model_name(name):
    name = str(name).strip()

    aliases = {
        "depth_anything": "depth_anything_v2_metric",
        "depth_anything_v2": "depth_anything_v2_metric",
        "depth_anything_v2_metric": "depth_anything_v2_metric",
        "depth_pro": "depth_pro",
        "zoedepth": "zoedepth",
    }

    return aliases.get(name, name)


def short_depth_label(depth_model):
    depth_model = normalise_depth_model_name(depth_model)

    if depth_model == "depth_pro":
        return "Depth Pro"
    if depth_model == "zoedepth":
        return "ZoeDepth"
    if depth_model == "depth_anything_v2_metric":
        return "Depth Anything V2"

    return depth_model


def get_profile(depth_model):
    depth_model = normalise_depth_model_name(depth_model)
    return MODEL_CONFIG_PROFILES.get(
        depth_model,
        {
            "profile_name": f"{depth_model} / unknown config",
            "enable_plane_removal": "",
            "plane_distance_threshold": "",
            "object_above_plane_threshold": "",
            "plane_min_inlier_ratio": "",
            "plane_min_normal_z": "",
            "notes": "No default configuration profile was defined for this model",
        },
    )


def get_config_value(row, key):
    depth_model = normalise_depth_model_name(row.get("depth_model", ""))
    profile = get_profile(depth_model)

    if key == "profile_name":
        return profile.get("profile_name", "")

    if row_has_value(row, key):
        return str(row.get(key)).strip()

    if key in profile:
        return profile[key]

    if key in COMMON_CONFIG_DEFAULTS:
        return COMMON_CONFIG_DEFAULTS[key]

    return ""


def unique_config_value(rows, key):
    values = []

    for row in rows:
        value = get_config_value(row, key)

        if value == "" or value.lower() == "nan":
            continue

        if value not in values:
            values.append(value)

    if len(values) == 0:
        return ""

    if len(values) == 1:
        return values[0]

    return "; ".join(values)


def model_pair_name(row):
    depth_model = row.get("depth_model", "")
    grasp_model = row.get("grasp_model", "")

    if depth_model == "" and grasp_model == "":
        return ""

    return f"{short_depth_label(depth_model)} + {grasp_model}"


def format_number(value, decimals=1):
    if not np.isfinite(value):
        return ""

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    if abs(value) >= 100:
        return f"{value:.0f}"

    return f"{value:.{decimals}f}"


def add_value_labels(ax, bars, values, decimals=0):
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue

        height = bar.get_height()

        ax.annotate(
            format_number(value, decimals=decimals),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def get_stage_statistics(rows):
    used_keys = []
    used_labels = []
    means = []
    stds = []
    mins = []
    maxs = []

    for key, label in zip(STAGE_KEYS, STAGE_LABELS):
        vals = valid_values(rows, key)

        if len(vals) == 0:
            continue

        used_keys.append(key)
        used_labels.append(label)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))

    return used_keys, used_labels, means, stds, mins, maxs


def plot_metric_per_trial(rows, key, ylabel, title, filename, decimals=1):
    values = np.array([to_float(r, key) for r in rows], dtype=float)
    valid = np.isfinite(values)

    x = np.arange(1, len(values) + 1)[valid]
    y = values[valid]

    if len(y) == 0:
        print(f"[WARN] No valid values for {key}. Skipping {filename}.")
        return

    tick_labels = []
    for i, row in enumerate(rows, start=1):
        if not valid[i - 1]:
            continue
        tick_labels.append(f"{i}\n{short_depth_label(row.get('depth_model', ''))}")

    plt.figure(figsize=(max(10, len(y) * 0.65), 5.2))
    ax = plt.gca()

    bars = ax.bar(x, y)
    add_value_labels(ax, bars, y, decimals=decimals)

    ax.set_xlabel("Trial and depth model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)

    y_max = np.max(y)
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.22)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=250)
    plt.close()


def plot_metric_mean_by_model(rows, key, ylabel, title, filename, decimals=1):
    grouped = {}

    for row in rows:
        depth_model = normalise_depth_model_name(row.get("depth_model", ""))
        if depth_model == "":
            continue

        grouped.setdefault(depth_model, []).append(row)

    labels = []
    means = []
    stds = []

    for depth_model in sorted(grouped.keys()):
        vals = valid_values(grouped[depth_model], key)

        if len(vals) == 0:
            continue

        labels.append(short_depth_label(depth_model))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    if len(means) == 0:
        print(f"[WARN] No valid per-model values for {key}. Skipping {filename}.")
        return

    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)

    plt.figure(figsize=(8.5, 5.0))
    ax = plt.gca()

    bars = ax.bar(labels, means, yerr=stds, capsize=4)
    add_value_labels(ax, bars, means, decimals=decimals)

    ax.set_xlabel("Depth model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    y_max = np.max(means + stds)
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.25)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=250)
    plt.close()


def plot_mean_stage_counts(rows):
    _, used_labels, means, stds, _, _ = get_stage_statistics(rows)

    if len(means) == 0:
        print("[WARN] No valid stage point-count values. Skipping stage plots.")
        return

    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)

    positive = np.isfinite(means) & (means > 0)

    log_labels = [label for label, keep in zip(used_labels, positive) if keep]
    log_means = means[positive]
    log_stds = stds[positive]

    plt.figure(figsize=(11, 5.2))
    ax = plt.gca()

    bars = ax.bar(log_labels, log_means, yerr=log_stds, capsize=4)
    ax.set_xlabel("Processing stage")
    ax.set_ylabel("Number of points (log scale)")
    ax.set_title("Mean point count by processing stage, all models")
    ax.set_yscale("log")

    add_value_labels(ax, bars, log_means, decimals=0)

    if len(log_means) > 0:
        ax.set_ylim(1, np.max(log_means) * 5.0)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_point_count_by_stage_full_log.png", dpi=250)
    plt.close()

    zoom_labels = []
    zoom_means = []
    zoom_stds = []

    for label, mean, std in zip(used_labels, means, stds):
        if label in ["Global fused", "Candidate"]:
            continue

        if not np.isfinite(mean):
            continue

        zoom_labels.append(label)
        zoom_means.append(mean)
        zoom_stds.append(std)

    if len(zoom_means) == 0:
        print("[WARN] No valid zoom-stage values. Skipping zoom plot.")
        return

    zoom_means = np.array(zoom_means, dtype=float)
    zoom_stds = np.array(zoom_stds, dtype=float)

    plt.figure(figsize=(11, 5.2))
    ax = plt.gca()

    bars = ax.bar(zoom_labels, zoom_means, yerr=zoom_stds, capsize=4)
    ax.set_xlabel("Processing stage")
    ax.set_ylabel("Number of points")
    ax.set_title("Mean point count by cleaning stage, all models")

    add_value_labels(ax, bars, zoom_means, decimals=0)

    y_max = np.max(zoom_means + zoom_stds)
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.25)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_point_count_by_stage_zoom.png", dpi=250)
    plt.close()


def plot_retention_percentage(rows):
    global_values = np.array([to_float(r, "stage_00_global_fused") for r in rows], dtype=float)

    means = []
    stds = []
    labels = []

    for key, label in zip(STAGE_KEYS, STAGE_LABELS):
        stage_values = np.array([to_float(r, key) for r in rows], dtype=float)

        valid = (
            np.isfinite(global_values)
            & np.isfinite(stage_values)
            & (global_values > 0)
            & (stage_values >= 0)
        )

        if np.count_nonzero(valid) == 0:
            continue

        retention = stage_values[valid] / global_values[valid] * 100.0

        means.append(float(np.mean(retention)))
        stds.append(float(np.std(retention)))
        labels.append(label)

    if len(means) == 0:
        print("[WARN] No valid retention values. Skipping retention plot.")
        return

    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)

    plt.figure(figsize=(11, 5.2))
    ax = plt.gca()

    bars = ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set_xlabel("Processing stage")
    ax.set_ylabel("Retained points (%)")
    ax.set_title("Mean point retention by processing stage, all models")

    add_value_labels(ax, bars, means, decimals=1)

    y_max = np.max(means + stds)
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, min(110.0, y_max * 1.25))

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "point_retention_percentage_by_stage.png", dpi=250)
    plt.close()


def write_summary_csv(rows):
    out_path = OUT_DIR / "summary_statistics.csv"

    depth_models = sorted(set(short_depth_label(r.get("depth_model", "")) for r in rows if r.get("depth_model", "") != ""))
    grasp_models = sorted(set(r.get("grasp_model", "") for r in rows if r.get("grasp_model", "") != ""))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_models_included", "grasp_models_included"])
        writer.writerow(["; ".join(depth_models), "; ".join(grasp_models)])
        writer.writerow([])
        writer.writerow(["metric", "n", "mean", "std", "min", "max"])

        for key, label in SUMMARY_METRICS:
            vals = valid_values(rows, key)

            if len(vals) == 0:
                writer.writerow([label, 0, "", "", "", ""])
                continue

            writer.writerow([
                label,
                len(vals),
                f"{np.mean(vals):.4f}",
                f"{np.std(vals):.4f}",
                f"{np.min(vals):.4f}",
                f"{np.max(vals):.4f}",
            ])

    return out_path


def write_model_summary_csv(rows):
    out_path = OUT_DIR / "per_model_summary_statistics.csv"

    grouped = {}
    for row in rows:
        key = (
            normalise_depth_model_name(row.get("depth_model", "unknown_depth")),
            row.get("grasp_model", "unknown_grasp"),
        )
        grouped.setdefault(key, []).append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "depth_model",
            "grasp_model",
            "configuration_profile",
            "plane_removal",
            "metric",
            "n",
            "mean",
            "std",
            "min",
            "max",
        ])

        for (depth_model, grasp_model), group_rows in sorted(grouped.items()):
            profile = get_profile(depth_model)
            plane_removal = unique_config_value(group_rows, "enable_plane_removal")

            for metric_key, metric_label in SUMMARY_METRICS:
                vals = valid_values(group_rows, metric_key)

                if len(vals) == 0:
                    writer.writerow([
                        short_depth_label(depth_model),
                        grasp_model,
                        profile.get("profile_name", ""),
                        plane_removal,
                        metric_label,
                        0,
                        "",
                        "",
                        "",
                        "",
                    ])
                    continue

                writer.writerow([
                    short_depth_label(depth_model),
                    grasp_model,
                    profile.get("profile_name", ""),
                    plane_removal,
                    metric_label,
                    len(vals),
                    f"{np.mean(vals):.4f}",
                    f"{np.std(vals):.4f}",
                    f"{np.min(vals):.4f}",
                    f"{np.max(vals):.4f}",
                ])

    return out_path


def write_model_configuration_csv(rows):
    out_path = OUT_DIR / "model_configuration_table.csv"

    grouped = {}
    for row in rows:
        depth_model = normalise_depth_model_name(row.get("depth_model", "unknown_depth"))
        grasp_model = row.get("grasp_model", get_config_value(row, "grasp_model"))
        grouped.setdefault((depth_model, grasp_model), []).append(row)

    columns = [
        "depth_model",
        "grasp_model",
        "configuration_profile",
        "notes",
    ] + [label for _, label in PER_TRIAL_CONFIG_COLUMNS if label != "Configuration profile"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for (depth_model, grasp_model), group_rows in sorted(grouped.items()):
            profile = get_profile(depth_model)

            output_row = [
                short_depth_label(depth_model),
                grasp_model,
                profile.get("profile_name", ""),
                profile.get("notes", ""),
            ]

            for key, label in PER_TRIAL_CONFIG_COLUMNS:
                if key == "profile_name":
                    continue
                output_row.append(unique_config_value(group_rows, key))

            writer.writerow(output_row)

    return out_path


def write_model_configuration_markdown(rows):
    out_path = OUT_DIR / "model_configuration_table.md"

    grouped = {}
    for row in rows:
        depth_model = normalise_depth_model_name(row.get("depth_model", "unknown_depth"))
        grasp_model = row.get("grasp_model", get_config_value(row, "grasp_model"))
        grouped.setdefault((depth_model, grasp_model), []).append(row)

    columns = [
        ("depth_model", "Depth model"),
        ("grasp_model", "Grasp model"),
        ("profile_name", "Profile"),
        ("enable_plane_removal", "Plane removal"),
        ("plane_distance_threshold", "Plane dist."),
        ("object_above_plane_threshold", "Above-plane th."),
        ("plane_min_normal_z", "Plane min normal-z"),
        ("dbscan_eps", "DBSCAN eps"),
        ("notes", "Notes"),
    ]

    with open(out_path, "w") as f:
        f.write("| " + " | ".join([name for _, name in columns]) + " |\n")
        f.write("| " + " | ".join(["---" for _ in columns]) + " |\n")

        for (depth_model, grasp_model), group_rows in sorted(grouped.items()):
            profile = get_profile(depth_model)
            values = []

            for key, _ in columns:
                if key == "depth_model":
                    values.append(short_depth_label(depth_model))
                elif key == "grasp_model":
                    values.append(grasp_model)
                elif key == "notes":
                    values.append(profile.get("notes", ""))
                elif key == "profile_name":
                    values.append(profile.get("profile_name", ""))
                else:
                    values.append(unique_config_value(group_rows, key))

            values = [str(v).replace("|", "/") for v in values]
            f.write("| " + " | ".join(values) + " |\n")

    return out_path


def write_stage_statistics_csv(rows):
    out_path = OUT_DIR / "stage_point_count_statistics.csv"
    global_values = np.array([to_float(r, "stage_00_global_fused") for r in rows], dtype=float)

    depth_models = sorted(set(short_depth_label(r.get("depth_model", "")) for r in rows if r.get("depth_model", "") != ""))
    grasp_models = sorted(set(r.get("grasp_model", "") for r in rows if r.get("grasp_model", "") != ""))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_models_included", "grasp_models_included"])
        writer.writerow(["; ".join(depth_models), "; ".join(grasp_models)])
        writer.writerow([])
        writer.writerow([
            "stage",
            "n",
            "mean_points",
            "std_points",
            "min_points",
            "max_points",
            "mean_retention_percent",
            "std_retention_percent",
        ])

        for key, label in zip(STAGE_KEYS, STAGE_LABELS):
            stage_values = np.array([to_float(r, key) for r in rows], dtype=float)
            valid_stage = np.isfinite(stage_values)

            if np.count_nonzero(valid_stage) == 0:
                writer.writerow([label, 0, "", "", "", "", "", ""])
                continue

            vals = stage_values[valid_stage]

            valid_retention = (
                np.isfinite(global_values)
                & np.isfinite(stage_values)
                & (global_values > 0)
                & (stage_values >= 0)
            )

            if np.count_nonzero(valid_retention) > 0:
                retention = stage_values[valid_retention] / global_values[valid_retention] * 100.0
                mean_retention = np.mean(retention)
                std_retention = np.std(retention)
            else:
                mean_retention = np.nan
                std_retention = np.nan

            writer.writerow([
                label,
                len(vals),
                f"{np.mean(vals):.4f}",
                f"{np.std(vals):.4f}",
                f"{np.min(vals):.4f}",
                f"{np.max(vals):.4f}",
                f"{mean_retention:.4f}" if np.isfinite(mean_retention) else "",
                f"{std_retention:.4f}" if np.isfinite(std_retention) else "",
            ])

    return out_path


def write_model_stage_statistics_csv(rows):
    out_path = OUT_DIR / "per_model_stage_point_count_statistics.csv"

    grouped = {}
    for row in rows:
        depth_model = normalise_depth_model_name(row.get("depth_model", "unknown_depth"))
        grasp_model = row.get("grasp_model", "unknown_grasp")
        grouped.setdefault((depth_model, grasp_model), []).append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "depth_model",
            "grasp_model",
            "configuration_profile",
            "stage",
            "n",
            "mean_points",
            "std_points",
            "min_points",
            "max_points",
        ])

        for (depth_model, grasp_model), group_rows in sorted(grouped.items()):
            profile = get_profile(depth_model)

            for key, label in zip(STAGE_KEYS, STAGE_LABELS):
                vals = valid_values(group_rows, key)

                if len(vals) == 0:
                    writer.writerow([
                        short_depth_label(depth_model),
                        grasp_model,
                        profile.get("profile_name", ""),
                        label,
                        0,
                        "",
                        "",
                        "",
                        "",
                    ])
                    continue

                writer.writerow([
                    short_depth_label(depth_model),
                    grasp_model,
                    profile.get("profile_name", ""),
                    label,
                    len(vals),
                    f"{np.mean(vals):.4f}",
                    f"{np.std(vals):.4f}",
                    f"{np.min(vals):.4f}",
                    f"{np.max(vals):.4f}",
                ])

    return out_path


def write_per_trial_results_csv(rows):
    out_path = OUT_DIR / "per_trial_results.csv"

    result_columns = [
        ("trial_id", "Trial ID"),
        ("depth_model", "Depth model"),
        ("grasp_model", "Grasp model"),
        ("gt_object_name", "Object"),
        ("center_error_xy_mm", "Centre XY error (mm)"),
        ("center_error_3d_mm", "Centre 3D error (mm)"),
        ("top_z_error_mm", "Top-z error (mm)"),
        ("yaw_error_deg", "Yaw error (deg)"),
        ("cloud_est_to_gt_mean_mm", "Cloud est-to-GT mean (mm)"),
        ("cloud_est_to_gt_median_mm", "Cloud est-to-GT median (mm)"),
        ("cloud_est_to_gt_p95_mm", "Cloud est-to-GT p95 (mm)"),
        ("cloud_gt_to_est_mean_mm", "Cloud GT-to-est mean (mm)"),
        ("cloud_gt_to_est_median_mm", "Cloud GT-to-est median (mm)"),
        ("cloud_gt_to_est_p95_mm", "Cloud GT-to-est p95 (mm)"),
        ("final_object_points", "Final points"),
        ("total_compute_time_s", "Compute time (s)"),
    ]

    columns = result_columns + PER_TRIAL_CONFIG_COLUMNS

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([display_name for _, display_name in columns])

        for row in rows:
            output_row = []

            for key, _ in columns:
                if key in ["trial_id", "gt_object_name"]:
                    output_row.append(row.get(key, ""))
                elif key == "depth_model":
                    output_row.append(short_depth_label(row.get(key, "")))
                elif key == "grasp_model":
                    output_row.append(row.get(key, get_config_value(row, key)))
                elif key in [k for k, _ in PER_TRIAL_CONFIG_COLUMNS]:
                    output_row.append(get_config_value(row, key))
                else:
                    value = to_float(row, key)

                    if not np.isfinite(value):
                        output_row.append("")
                    elif key == "final_object_points":
                        output_row.append(f"{value:.0f}")
                    elif key == "total_compute_time_s":
                        output_row.append(f"{value:.3f}")
                    else:
                        output_row.append(f"{value:.2f}")

            writer.writerow(output_row)

    return out_path


def write_per_trial_results_markdown(rows):
    out_path = OUT_DIR / "per_trial_results_table.md"

    columns = [
        ("trial_id", "Trial"),
        ("depth_model", "Depth model"),
        ("profile_name", "Profile"),
        ("enable_plane_removal", "Plane"),
        ("gt_object_name", "Object"),
        ("center_error_xy_mm", "XY error (mm)"),
        ("top_z_error_mm", "Top-z error (mm)"),
        ("yaw_error_deg", "Yaw error (deg)"),
        ("cloud_est_to_gt_mean_mm", "Cloud mean (mm)"),
        ("cloud_est_to_gt_p95_mm", "Cloud p95 (mm)"),
        ("final_object_points", "Final points"),
        ("total_compute_time_s", "Time (s)"),
    ]

    with open(out_path, "w") as f:
        f.write("| " + " | ".join([name for _, name in columns]) + " |\n")
        f.write("| " + " | ".join(["---" for _ in columns]) + " |\n")

        for i, row in enumerate(rows, start=1):
            values = []

            for key, _ in columns:
                if key == "trial_id":
                    values.append(str(i))
                elif key == "depth_model":
                    values.append(short_depth_label(row.get(key, "")))
                elif key in ["profile_name", "enable_plane_removal"]:
                    values.append(get_config_value(row, key))
                elif key == "gt_object_name":
                    values.append(row.get(key, ""))
                else:
                    value = to_float(row, key)

                    if not np.isfinite(value):
                        values.append("")
                    elif key == "final_object_points":
                        values.append(f"{value:.0f}")
                    elif key == "total_compute_time_s":
                        values.append(f"{value:.3f}")
                    else:
                        values.append(f"{value:.2f}")

            values = [str(v).replace("|", "/") for v in values]
            f.write("| " + " | ".join(values) + " |\n")

    return out_path


def print_summary(rows):
    metrics = [
        ("center_error_xy_mm", "Centre XY error (mm)"),
        ("center_error_3d_mm", "Centre 3D error (mm)"),
        ("top_z_error_mm", "Top height error (mm)"),
        ("yaw_error_deg", "Yaw error (deg)"),
        ("cloud_est_to_gt_mean_mm", "Cloud est-to-GT mean distance (mm)"),
        ("cloud_est_to_gt_median_mm", "Cloud est-to-GT median distance (mm)"),
        ("cloud_est_to_gt_p95_mm", "Cloud est-to-GT p95 distance (mm)"),
        ("final_object_points", "Final object points"),
        ("total_compute_time_s", "Total compute time (s)"),
    ]

    print("\n=== Evaluation Summary ===")

    model_pairs = sorted(set(model_pair_name(r) for r in rows if model_pair_name(r) != ""))
    if len(model_pairs) > 0:
        print("Model pairs included: " + "; ".join(model_pairs))

    print("\n=== Configuration Profiles ===")
    for depth_model in sorted(set(normalise_depth_model_name(r.get("depth_model", "")) for r in rows if r.get("depth_model", "") != "")):
        profile = get_profile(depth_model)
        print(f"{short_depth_label(depth_model)}: {profile.get('profile_name', '')} | {profile.get('notes', '')}")

    print("\n=== Metrics ===")
    for key, label in metrics:
        vals = valid_values(rows, key)

        if len(vals) == 0:
            print(f"{label}: no valid values")
            continue

        print(
            f"{label}: n={len(vals)}, "
            f"mean={np.mean(vals):.3f}, "
            f"std={np.std(vals):.3f}, "
            f"min={np.min(vals):.3f}, "
            f"max={np.max(vals):.3f}"
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find {CSV_PATH}")

    rows = read_rows(CSV_PATH)

    if len(rows) == 0:
        raise RuntimeError("No rows in experiment log.")

    # Per-trial metric plots. X-axis labels include model names.
    plot_metric_per_trial(rows, "center_error_xy_mm", "Error (mm)", "Object centre XY error per trial by depth model", "centre_xy_error_per_trial.png", decimals=1)
    plot_metric_per_trial(rows, "center_error_3d_mm", "Error (mm)", "Object centre 3D error per trial by depth model", "centre_3d_error_per_trial.png", decimals=1)
    plot_metric_per_trial(rows, "top_z_error_mm", "Error (mm)", "Object top height error per trial by depth model", "top_z_error_per_trial.png", decimals=1)
    plot_metric_per_trial(rows, "yaw_error_deg", "Error (degrees)", "Grasp yaw error per trial by depth model", "yaw_error_per_trial.png", decimals=1)

    plot_metric_per_trial(rows, "cloud_est_to_gt_mean_mm", "Distance (mm)", "Estimated cloud to GT cloud mean distance per trial by depth model", "cloud_est_to_gt_mean_per_trial.png", decimals=1)
    plot_metric_per_trial(rows, "cloud_est_to_gt_median_mm", "Distance (mm)", "Estimated cloud to GT cloud median distance per trial by depth model", "cloud_est_to_gt_median_per_trial.png", decimals=1)
    plot_metric_per_trial(rows, "cloud_est_to_gt_p95_mm", "Distance (mm)", "Estimated cloud to GT cloud 95th percentile distance per trial by depth model", "cloud_est_to_gt_p95_per_trial.png", decimals=1)
    plot_metric_per_trial(rows, "cloud_gt_to_est_mean_mm", "Distance (mm)", "GT cloud to estimated cloud mean distance per trial by depth model", "cloud_gt_to_est_mean_per_trial.png", decimals=1)

    plot_metric_per_trial(rows, "total_compute_time_s", "Time (s)", "Total computation time per trial by depth model", "compute_time_per_trial.png", decimals=2)
    plot_metric_per_trial(rows, "final_object_points", "Number of points", "Final object point count per trial by depth model", "final_object_points_per_trial.png", decimals=0)

    # Per-model summary plots.
    plot_metric_mean_by_model(rows, "cloud_est_to_gt_mean_mm", "Distance (mm)", "Mean estimated-to-GT cloud distance by depth model", "model_mean_cloud_est_to_gt_mean.png", decimals=1)
    plot_metric_mean_by_model(rows, "cloud_est_to_gt_p95_mm", "Distance (mm)", "95th percentile estimated-to-GT cloud distance by depth model", "model_mean_cloud_est_to_gt_p95.png", decimals=1)
    plot_metric_mean_by_model(rows, "center_error_xy_mm", "Error (mm)", "Mean object centre XY error by depth model", "model_mean_centre_xy_error.png", decimals=1)
    plot_metric_mean_by_model(rows, "top_z_error_mm", "Error (mm)", "Mean object top height error by depth model", "model_mean_top_z_error.png", decimals=1)
    plot_metric_mean_by_model(rows, "total_compute_time_s", "Time (s)", "Mean computation time by depth model", "model_mean_compute_time.png", decimals=2)

    plot_mean_stage_counts(rows)
    plot_retention_percentage(rows)

    summary_csv = write_summary_csv(rows)
    model_summary_csv = write_model_summary_csv(rows)
    model_config_csv = write_model_configuration_csv(rows)
    model_config_md = write_model_configuration_markdown(rows)
    stage_csv = write_stage_statistics_csv(rows)
    model_stage_csv = write_model_stage_statistics_csv(rows)
    per_trial_csv = write_per_trial_results_csv(rows)
    per_trial_md = write_per_trial_results_markdown(rows)

    print_summary(rows)

    print(f"\nSaved summary figures to: {OUT_DIR}")
    print(f"Saved summary statistics to: {summary_csv}")
    print(f"Saved per-model summary statistics to: {model_summary_csv}")
    print(f"Saved model configuration table to: {model_config_csv}")
    print(f"Saved model configuration markdown table to: {model_config_md}")
    print(f"Saved stage point-count statistics to: {stage_csv}")
    print(f"Saved per-model stage statistics to: {model_stage_csv}")
    print(f"Saved per-trial results table to: {per_trial_csv}")
    print(f"Saved per-trial markdown table to: {per_trial_md}")


if __name__ == "__main__":
    main()