import json
import os

import numpy as np

from utils.config import Config


def save_result_artifacts(run_dir: str, results: dict):
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    summary = {
        "metrics": results.get("metrics", {}),
        "threshold": results.get("threshold"),
        "score_name": results.get("score_name", ""),
    }
    score_families = {}
    for key in ["center_distance", "reconstruction", "cross_view"]:
        family = results.get(key, {})
        if family:
            score_families[key] = {
                "metrics": family.get("metrics", {}),
                "threshold": family.get("threshold"),
            }
    for key, family in results.get("component_families", {}).items():
        if family:
            score_families[key] = {
                "metrics": family.get("metrics", {}),
                "threshold": family.get("threshold"),
            }
    if score_families:
        summary["score_families"] = score_families
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    npy_payloads = {
        "y_true.npy": results.get("y_true"),
        "test_latent_features.npy": results.get("test_features"),
    }
    if results.get("scores") is not None:
        npy_payloads["scores.npy"] = results.get("scores")
    if results.get("pred_labels") is not None:
        npy_payloads["pred_labels.npy"] = results.get("pred_labels")
    for key in ["center_distance", "reconstruction", "cross_view"]:
        family = results.get(key, {})
        if not family:
            continue
        npy_payloads[f"{key}_train_scores.npy"] = family.get("train_scores")
        npy_payloads[f"{key}_test_scores.npy"] = family.get("test_scores")
        npy_payloads[f"{key}_pred_labels.npy"] = family.get("pred_labels")
    for key, family in results.get("component_families", {}).items():
        if not family:
            continue
        npy_payloads[f"{key}_train_scores.npy"] = family.get("train_scores")
        npy_payloads[f"{key}_test_scores.npy"] = family.get("test_scores")
        npy_payloads[f"{key}_pred_labels.npy"] = family.get("pred_labels")
    for component_key, value in results.get("components", {}).items():
        if value is not None:
            npy_payloads[f"component_{component_key}.npy"] = value

    for filename, value in npy_payloads.items():
        if value is None:
            continue
        path = os.path.join(run_dir, filename)
        with open(path, "wb") as file:
            np.save(file, value)

    distance_analysis = results.get("distance_analysis", {})
    if distance_analysis:
        summary_path = os.path.join(run_dir, "test_latent_distance_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(distance_analysis.get("summary", {}), file, ensure_ascii=False, indent=2)

        summary = distance_analysis.get("summary", {})
        report_lines = ["Test Latent Distance Summary"]
        for key, title in [
            ("normal_normal", "Normal-Normal"),
            ("normal_anomaly", "Normal-Anomaly"),
            ("anomaly_anomaly", "Anomaly-Anomaly"),
        ]:
            item = summary.get(key, {})
            report_lines.append(
                (
                    f"{title}: "
                    f"pairs={int(item.get('pair_count', 0))}, "
                    f"mean={float(item.get('mean', 0.0)):.6f}, "
                    f"std={float(item.get('std', 0.0)):.6f}, "
                    f"min={float(item.get('min', 0.0)):.6f}, "
                    f"max={float(item.get('max', 0.0)):.6f}, "
                    f"sample_median={float(item.get('sample_median', 0.0)):.6f}"
                )
            )
        report_path = os.path.join(run_dir, "test_latent_distance_report.txt")
        with open(report_path, "w", encoding="utf-8") as file:
            file.write("\n".join(report_lines) + "\n")

        distance_arrays = {
            "test_latent_distance_indices.npy": distance_analysis.get("analyzed_indices"),
            "test_latent_distance_labels.npy": distance_analysis.get("analyzed_labels"),
            "test_latent_distance_features.npy": distance_analysis.get("analyzed_features"),
            "test_latent_distance_normal_normal_sample.npy": distance_analysis.get("samples", {}).get("normal_normal"),
            "test_latent_distance_normal_anomaly_sample.npy": distance_analysis.get("samples", {}).get("normal_anomaly"),
            "test_latent_distance_anomaly_anomaly_sample.npy": distance_analysis.get("samples", {}).get("anomaly_anomaly"),
        }
        for filename, value in distance_arrays.items():
            if value is None:
                continue
            path = os.path.join(run_dir, filename)
            with open(path, "wb") as file:
                np.save(file, value)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5.2))
            series = [
                ("normal_normal", "Normal-Normal", "#2563eb"),
                ("normal_anomaly", "Normal-Anomaly", "#ea580c"),
                ("anomaly_anomaly", "Anomaly-Anomaly", "#c026d3"),
            ]
            has_data = False
            for key, label, color in series:
                values = np.asarray(distance_analysis.get("samples", {}).get(key, []), dtype=np.float32).reshape(-1)
                if values.size == 0:
                    continue
                has_data = True
                ax.hist(
                    values,
                    bins=60,
                    density=True,
                    alpha=0.35,
                    color=color,
                    label=f"{label} (sample={int(values.size)})",
                )
            if has_data:
                ax.set_title("Test Latent Distance Distribution")
                ax.set_xlabel("L2 Distance in Latent Space")
                ax.set_ylabel("Density")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(run_dir, "test_latent_distance_distribution.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
            plt.close(fig)

            summary = distance_analysis.get("summary", {})
            plot_series = [
                ("normal_normal", "Normal-Normal", "#2563eb"),
                ("normal_anomaly", "Normal-Anomaly", "#ea580c"),
                ("anomaly_anomaly", "Anomaly-Anomaly", "#c026d3"),
            ]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
            boxplot_values = []
            boxplot_labels = []
            boxplot_colors = []
            for key, label, color in plot_series:
                values = np.asarray(distance_analysis.get("samples", {}).get(key, []), dtype=np.float32).reshape(-1)
                if values.size == 0:
                    continue
                boxplot_values.append(values)
                boxplot_labels.append(label)
                boxplot_colors.append(color)

            if boxplot_values:
                boxplot = axes[0].boxplot(
                    boxplot_values,
                    labels=boxplot_labels,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, color in zip(boxplot["boxes"], boxplot_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.35)
                axes[0].set_title("Sampled Latent Distance Boxplot")
                axes[0].set_ylabel("L2 Distance")
                axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
                for tick in axes[0].get_xticklabels():
                    tick.set_rotation(12)
            else:
                axes[0].text(0.5, 0.5, "No sampled distances", ha="center", va="center", fontsize=11)
                axes[0].set_axis_off()

            means = [float(summary.get(key, {}).get("mean", 0.0)) for key, _, _ in plot_series]
            medians = [float(summary.get(key, {}).get("sample_median", 0.0)) for key, _, _ in plot_series]
            positions = np.arange(len(plot_series), dtype=np.float32)
            width = 0.34
            axes[1].bar(
                positions - width / 2.0,
                means,
                width=width,
                color=[color for _, _, color in plot_series],
                alpha=0.75,
                label="Mean",
            )
            axes[1].bar(
                positions + width / 2.0,
                medians,
                width=width,
                color=[color for _, _, color in plot_series],
                alpha=0.35,
                label="Sample Median",
            )
            axes[1].set_xticks(positions)
            axes[1].set_xticklabels([label for _, label, _ in plot_series], rotation=12)
            axes[1].set_title("Latent Distance Summary")
            axes[1].set_ylabel("Distance")
            axes[1].legend(loc="best", fontsize=8)
            axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
            for idx, value in enumerate(means):
                axes[1].text(idx - width / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
            for idx, value in enumerate(medians):
                axes[1].text(idx + width / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

            fig.tight_layout()
            fig.savefig(
                os.path.join(run_dir, "test_latent_distance_summary_plot.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as exc:
            print(f"[Distance] Skip latent distance plot: {type(exc).__name__}: {exc}")

    banks = results.get("banks", {})
    if not banks:
        return

    bank_arrays = {
        "prototype_centers.npy": banks.get("prototype_centers"),
        "global_center.npy": banks.get("global_center"),
        "core_mask.npy": banks.get("core_mask"),
        "score_core.npy": banks.get("score_core"),
        "cluster_labels.npy": banks.get("cluster_labels"),
        "cluster_core_radius.npy": banks.get("cluster_core_radius"),
        "cluster_scoring_radius.npy": banks.get("cluster_scoring_radius"),
        "nearest_other_cluster.npy": banks.get("nearest_other_cluster"),
        "proto_conf1.npy": banks.get("proto_conf1"),
        "proto_conf2.npy": banks.get("proto_conf2"),
        "proto_pred1.npy": banks.get("proto_pred1"),
        "proto_pred2.npy": banks.get("proto_pred2"),
        "proto_dist1.npy": banks.get("proto_dist1"),
        "proto_dist2.npy": banks.get("proto_dist2"),
        "cluster_radii_v1.npy": banks.get("cluster_radii_v1"),
        "cluster_radii_v2.npy": banks.get("cluster_radii_v2"),
        "local_radius_v1.npy": banks.get("local_radius_v1"),
        "local_radius_v2.npy": banks.get("local_radius_v2"),
        "local_dist_v1.npy": banks.get("local_dist_v1"),
        "local_dist_v2.npy": banks.get("local_dist_v2"),
        "proto_recon1.npy": banks.get("recon1"),
        "proto_recon2.npy": banks.get("recon2"),
    }
    for filename, value in bank_arrays.items():
        if value is None:
            continue
        path = os.path.join(run_dir, filename)
        with open(path, "wb") as file:
            np.save(file, value)

    bank_summary = banks.get("summary", [])
    if bank_summary:
        summary_path = os.path.join(run_dir, "bank_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(bank_summary, file, ensure_ascii=False, indent=2)

    joint_core_label_diagnostics = banks.get("joint_core_label_diagnostics", [])
    if joint_core_label_diagnostics:
        diagnostics_path = os.path.join(run_dir, "joint_core_label_diagnostics.json")
        with open(diagnostics_path, "w", encoding="utf-8") as file:
            json.dump(joint_core_label_diagnostics, file, ensure_ascii=False, indent=2)

def save_config_artifact(run_dir: str, config: Config):
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, ensure_ascii=False, indent=2)
