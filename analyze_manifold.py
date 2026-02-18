#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Manifold + hysteresis analysis from Bob JSONL traces.

Two claims separate Bob from "bandit bolted onto Mixtral":
1. Routing trajectory collapses onto a low-dimensional manifold
2. Scars create hysteresis loops and persistent deformation

Usage:
    # Manifold comparison (governed vs baseline)
    python3 analyze_manifold.py governed.jsonl baseline.jsonl

    # Hysteresis analysis (single run with perturb_phase)
    python3 analyze_manifold.py --hysteresis governed.jsonl

    # Both
    python3 analyze_manifold.py --hysteresis governed.jsonl baseline.jsonl

Dependencies: numpy, sklearn (PCA), matplotlib. UMAP optional.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_traces(jsonl_path: str) -> Tuple[Optional[Dict], List[Dict]]:
    """Load JSONL file, return (header, traces)."""
    header = None
    traces = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                header = record
                continue
            traces.append(record)
    return header, traces


def extract_feature_vector(trace: Dict) -> Optional[List[float]]:
    """Extract geometry feature vector from a single trace.

    Features:
        [router_entropy, churn, flipflop_ema, medium_activation,
         scar_debt, loss, neff, expert_id_0/64, expert_id_1/64]
    """
    # Require geometry fields
    if trace.get("router_entropy") is None:
        return None

    router_entropy = trace.get("router_entropy", 0.0)
    churn = trace.get("churn", 0.0)
    flipflop_ema = trace.get("flipflop_ema", 0.0)
    medium_activation = trace.get("medium_activation", 0.0)
    scar_debt = trace.get("scar_debt", 0.0)
    loss = trace.get("loss", 0.0)
    neff = trace.get("neff", 0.0)

    # Normalized expert hash features from first two expert IDs
    expert_ids = trace.get("expert_ids", [])
    eid0 = expert_ids[0] / 64.0 if len(expert_ids) > 0 else 0.0
    eid1 = expert_ids[1] / 64.0 if len(expert_ids) > 1 else 0.0

    return [
        router_entropy, churn, flipflop_ema, medium_activation,
        scar_debt, loss, neff, eid0, eid1,
    ]


FEATURE_NAMES = [
    "router_entropy", "churn", "flipflop_ema", "medium_activation",
    "scar_debt", "loss", "neff", "expert_id_0", "expert_id_1",
]

# Named feature subsets for ablation
FEATURE_PRESETS = {
    "all": None,  # Use all features
    "routing_only": ["router_entropy", "churn", "flipflop_ema", "neff"],
    "routing_plus_loss": ["router_entropy", "churn", "flipflop_ema", "neff", "loss"],
    "governance": ["medium_activation", "scar_debt", "flipflop_ema"],
}


def _feature_indices(preset: Optional[str]) -> Optional[List[int]]:
    """Return column indices for a feature preset, or None for all."""
    if preset is None or preset == "all":
        return None
    names = FEATURE_PRESETS.get(preset)
    if names is None:
        raise ValueError(f"Unknown feature preset: {preset}. "
                         f"Available: {list(FEATURE_PRESETS.keys())}")
    return [FEATURE_NAMES.index(n) for n in names]


def _subset_features(features: np.ndarray, preset: Optional[str]) -> Tuple[np.ndarray, List[str]]:
    """Subset feature matrix by preset. Returns (subsetted_matrix, feature_names)."""
    indices = _feature_indices(preset)
    if indices is None:
        return features, FEATURE_NAMES
    return features[:, indices], [FEATURE_NAMES[i] for i in indices]


def extract_features(traces: List[Dict], skip_warmup: int = 0) -> np.ndarray:
    """Extract feature matrix from traces, skipping warmup steps."""
    vectors = []
    for t in traces:
        if t["step"] < skip_warmup:
            continue
        vec = extract_feature_vector(t)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return np.zeros((0, len(FEATURE_NAMES)))
    return np.array(vectors)


def manifold_analysis(
    governed_path: str,
    baseline_path: str,
    output_dir: str = ".",
    skip_warmup: int = 0,
    feature_preset: Optional[str] = None,
) -> Dict:
    """PCA-based manifold comparison between governed and baseline runs."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gov_header, gov_traces = load_traces(governed_path)
    base_header, base_traces = load_traces(baseline_path)

    warmup = skip_warmup
    if warmup == 0 and gov_header:
        warmup = gov_header.get("warmup_steps", 0)

    gov_features_full = extract_features(gov_traces, skip_warmup=warmup)
    base_features_full = extract_features(base_traces, skip_warmup=warmup)

    if gov_features_full.shape[0] < 5 or base_features_full.shape[0] < 5:
        print("ERROR: Not enough traces with geometry fields "
              f"(governed={gov_features_full.shape[0]}, baseline={base_features_full.shape[0]})")
        return {"error": "insufficient_data"}

    # Apply feature subset
    gov_features, used_features = _subset_features(gov_features_full, feature_preset)
    base_features, _ = _subset_features(base_features_full, feature_preset)
    preset_label = feature_preset or "all"
    print(f"\nFeatures ({preset_label}): {used_features}")

    # Combine for joint PCA
    combined = np.vstack([gov_features, base_features])
    labels = np.array(
        [1] * gov_features.shape[0] + [0] * base_features.shape[0]
    )  # 1=governed, 0=baseline

    # Normalize
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    # PCA
    n_components = min(5, combined_scaled.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(combined_scaled)

    # Variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print("\nPCA cumulative variance explained:")
    for i, cv in enumerate(cumvar):
        print(f"  PC{i+1}: {cv:.4f} ({pca.explained_variance_ratio_[i]:.4f})")

    # Silhouette score
    sil = silhouette_score(projected[:, :2], labels)
    print(f"\nSilhouette score (governed vs baseline): {sil:.4f}")

    # --- Plot 1: PC1 vs PC2, color by governed/baseline ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    base_mask = labels == 0
    gov_mask = labels == 1
    ax.scatter(projected[base_mask, 0], projected[base_mask, 1],
               alpha=0.4, s=12, c="tab:blue", label="baseline")
    ax.scatter(projected[gov_mask, 0], projected[gov_mask, 1],
               alpha=0.4, s=12, c="tab:orange", label="governed")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"Routing Manifold: Gov vs Base [{preset_label}]")
    ax.legend()

    # --- Plot 2: PC1 vs PC2, color by category ---
    ax = axes[1]
    category_names = ["code", "math", "dialogue", "reasoning", "factual"]
    # Extract categories for all traces
    all_traces = []
    for t in gov_traces:
        if t["step"] >= warmup and extract_feature_vector(t) is not None:
            all_traces.append(t)
    for t in base_traces:
        if t["step"] >= warmup and extract_feature_vector(t) is not None:
            all_traces.append(t)

    categories = np.array([t.get("context_class", 0) for t in all_traces])
    colors = plt.cm.Set1(np.linspace(0, 1, len(category_names)))
    for ci, cname in enumerate(category_names):
        mask = categories == ci
        if mask.sum() > 0:
            ax.scatter(projected[mask, 0], projected[mask, 1],
                       alpha=0.4, s=12, c=[colors[ci]], label=cname)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Routing Manifold: By Category")
    ax.legend()

    plt.tight_layout()
    suffix = f"_{preset_label}" if preset_label != "all" else ""
    plot_path = os.path.join(output_dir, f"manifold_pca{suffix}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # Build results
    results = {
        "governed_samples": int(gov_features.shape[0]),
        "baseline_samples": int(base_features.shape[0]),
        "feature_preset": preset_label,
        "pca_variance_explained": [round(float(v), 4) for v in cumvar],
        "pca_individual": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "silhouette_score": round(float(sil), 4),
        "feature_names": used_features,
        "plot": plot_path,
    }

    # Component loadings
    loadings = {}
    for i in range(min(3, n_components)):
        loadings[f"PC{i+1}"] = {
            name: round(float(w), 4)
            for name, w in zip(used_features, pca.components_[i])
        }
    results["loadings"] = loadings

    return results


def hysteresis_analysis(
    jsonl_path: str,
    output_dir: str = ".",
    centroid_threshold: float = 0.5,
) -> Dict:
    """Analyze trajectory hysteresis from a run with perturb_phase annotations."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    header, traces = load_traces(jsonl_path)

    # Check for perturbation phases
    phases = set(t.get("perturb_phase") for t in traces if "perturb_phase" in t)
    if not phases:
        print("ERROR: No perturb_phase annotations found in JSONL. "
              "Run with --perturb-at-step to generate perturbation data.")
        return {"error": "no_perturb_phase"}

    # Split by phase
    before = [t for t in traces if t.get("perturb_phase") == "before"]
    during = [t for t in traces if t.get("perturb_phase") == "during"]
    after = [t for t in traces if t.get("perturb_phase") == "after"]

    print(f"Traces by phase: before={len(before)}, during={len(during)}, after={len(after)}")

    # Extract features for all phases
    all_phase_traces = before + during + after
    features = []
    valid_traces = []
    for t in all_phase_traces:
        vec = extract_feature_vector(t)
        if vec is not None:
            features.append(vec)
            valid_traces.append(t)

    if len(features) < 10:
        print(f"ERROR: Not enough traces with geometry fields ({len(features)})")
        return {"error": "insufficient_data"}

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=min(5, features_scaled.shape[1]))
    projected = pca.fit_transform(features_scaled)

    # Phase labels
    phase_labels = np.array([t.get("perturb_phase", "unknown") for t in valid_traces])

    # Centroids
    before_mask = phase_labels == "before"
    during_mask = phase_labels == "during"
    after_mask = phase_labels == "after"

    centroids = {}
    for name, mask in [("before", before_mask), ("during", during_mask), ("after", after_mask)]:
        if mask.sum() > 0:
            centroids[name] = projected[mask, :2].mean(axis=0)

    # Centroid distance: before vs after
    hysteresis_distance = None
    hysteresis_confirmed = False
    if "before" in centroids and "after" in centroids:
        hysteresis_distance = float(
            np.linalg.norm(centroids["after"] - centroids["before"])
        )
        hysteresis_confirmed = hysteresis_distance > centroid_threshold
        print(f"\nCentroid distance (before → after): {hysteresis_distance:.4f}")
        print(f"Threshold: {centroid_threshold}")
        print(f"Hysteresis: {'CONFIRMED' if hysteresis_confirmed else 'NOT CONFIRMED'}")

    # Loop area (shoelace formula on trajectory in PC1-PC2)
    loop_area = None
    if projected.shape[0] >= 3:
        x = projected[:, 0]
        y = projected[:, 1]
        # Shoelace formula for signed area of closed polygon
        loop_area = float(0.5 * abs(
            np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
            + x[-1] * y[0] - x[0] * y[-1]
        ))
        print(f"Loop area (PC1-PC2): {loop_area:.4f}")

    # --- Plot: Trajectory as time-ordered path ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter by phase
    ax = axes[0]
    phase_colors = {"before": "tab:blue", "during": "tab:red", "after": "tab:green"}
    for phase_name, color in phase_colors.items():
        mask = phase_labels == phase_name
        if mask.sum() > 0:
            ax.scatter(projected[mask, 0], projected[mask, 1],
                       alpha=0.5, s=15, c=color, label=phase_name)
    # Mark centroids
    for name, centroid in centroids.items():
        ax.scatter(centroid[0], centroid[1], marker="X", s=200,
                   c=phase_colors.get(name, "black"), edgecolors="black",
                   linewidths=1.5, zorder=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Hysteresis: Phase Scatter")
    ax.legend()

    # Right: time-ordered trajectory path
    ax = axes[1]
    # Color gradient by time
    steps = np.array([t["step"] for t in valid_traces])
    norm_steps = (steps - steps.min()) / max(steps.max() - steps.min(), 1)
    scatter = ax.scatter(projected[:, 0], projected[:, 1],
                         c=norm_steps, cmap="viridis", alpha=0.6, s=12)
    # Draw path
    ax.plot(projected[:, 0], projected[:, 1], color="gray", alpha=0.2, linewidth=0.5)
    # Mark perturbation boundaries
    perturb_start = None
    perturb_end = None
    for i, t in enumerate(valid_traces):
        if t.get("perturb_phase") == "during" and perturb_start is None:
            perturb_start = i
        if t.get("perturb_phase") == "after" and perturb_end is None:
            perturb_end = i
    if perturb_start is not None:
        ax.scatter(projected[perturb_start, 0], projected[perturb_start, 1],
                   marker="v", s=150, c="red", edgecolors="black",
                   linewidths=1.5, zorder=10, label="perturb start")
    if perturb_end is not None:
        ax.scatter(projected[perturb_end, 0], projected[perturb_end, 1],
                   marker="^", s=150, c="green", edgecolors="black",
                   linewidths=1.5, zorder=10, label="perturb end")
    plt.colorbar(scatter, ax=ax, label="time (normalized)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Hysteresis: Time Trajectory")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "hysteresis_trajectory.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    return {
        "traces_before": int(before_mask.sum()),
        "traces_during": int(during_mask.sum()),
        "traces_after": int(after_mask.sum()),
        "pca_variance_explained": [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)],
        "centroids": {k: [round(float(c), 4) for c in v] for k, v in centroids.items()},
        "hysteresis_distance": round(hysteresis_distance, 4) if hysteresis_distance is not None else None,
        "hysteresis_confirmed": hysteresis_confirmed,
        "centroid_threshold": centroid_threshold,
        "loop_area": round(loop_area, 4) if loop_area is not None else None,
        "plot": plot_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Manifold + hysteresis analysis from Bob JSONL traces"
    )
    parser.add_argument("jsonl_files", nargs="+",
                        help="JSONL files (1 for hysteresis, 2 for manifold comparison)")
    parser.add_argument("--hysteresis", action="store_true",
                        help="Run hysteresis analysis on first JSONL file")
    parser.add_argument("--skip-warmup", type=int, default=0,
                        help="Skip first N steps (0 = use header warmup_steps)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output plots and JSON")
    parser.add_argument("--centroid-threshold", type=float, default=0.5,
                        help="Distance threshold for hysteresis confirmation")
    parser.add_argument("--features", default=None,
                        help="Feature preset: all, routing_only, routing_plus_loss, governance")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # Manifold comparison (needs 2 files)
    if len(args.jsonl_files) >= 2:
        print("=" * 60)
        print("MANIFOLD ANALYSIS")
        print("=" * 60)
        results = manifold_analysis(
            governed_path=args.jsonl_files[0],
            baseline_path=args.jsonl_files[1],
            output_dir=args.output_dir,
            skip_warmup=args.skip_warmup,
            feature_preset=args.features,
        )
        all_results["manifold"] = results

    # Hysteresis analysis
    if args.hysteresis:
        print("\n" + "=" * 60)
        print("HYSTERESIS ANALYSIS")
        print("=" * 60)
        results = hysteresis_analysis(
            jsonl_path=args.jsonl_files[0],
            output_dir=args.output_dir,
            centroid_threshold=args.centroid_threshold,
        )
        all_results["hysteresis"] = results

    if not all_results:
        # Default: if only 1 file and no --hysteresis, try manifold with just that file
        print("Provide 2 JSONL files for manifold comparison, "
              "or use --hysteresis for single-file trajectory analysis.")
        sys.exit(1)

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "manifold_analysis.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
