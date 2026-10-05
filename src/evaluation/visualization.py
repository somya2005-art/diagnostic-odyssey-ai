"""Publication-Grade Plotting and Visualization Module.

Generates ROC Curves, Precision-Recall Curves, Calibration Curves,
and Benchmark Metric Comparisons for research papers and presentations.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve


class PublicationPlotter:
    """Generates clean, aesthetic figures formatted for academic papers."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.05)

    def plot_roc_curves(
        self,
        models_predictions: Dict[str, Dict[str, np.ndarray]],
        filename: str = "roc_curves_comparison.png"
    ):
        """Plots multi-model ROC curves on a single figure.
        
        Args:
            models_predictions: Dict mapping model_name -> {'y_true': np.ndarray, 'y_prob': np.ndarray}
            filename: Output filename.
        """
        plt.figure(figsize=(7.5, 6), dpi=300)
        palette = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c", "#16a085"]

        for idx, (name, data) in enumerate(models_predictions.items()):
            y_true, y_prob = data["y_true"], data["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_score = auc(fpr, tpr)
            color = palette[idx % len(palette)]
            linewidth = 2.5 if "BiLSTM" in name or "Transformer" in name else 1.8
            plt.plot(fpr, tpr, label=f"{name} (AUROC = {roc_score:.3f})", color=color, linewidth=linewidth)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline (AUROC = 0.500)")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
        plt.title("Diagnostic Delay Prediction: ROC Curve Comparison", fontsize=13, fontweight="bold", pad=12)
        plt.legend(loc="lower right", frameon=True, fontsize=9.5)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[Plotter] Saved ROC comparison figure to {out_path}")

    def plot_pr_curves(
        self,
        models_predictions: Dict[str, Dict[str, np.ndarray]],
        filename: str = "pr_curves_comparison.png"
    ):
        """Plots multi-model Precision-Recall curves."""
        plt.figure(figsize=(7.5, 6), dpi=300)
        palette = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c", "#16a085"]

        baseline_rate = 0.5
        for idx, (name, data) in enumerate(models_predictions.items()):
            y_true, y_prob = data["y_true"], data["y_prob"]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_score = auc(rec, prec)
            baseline_rate = float(np.mean(y_true))
            color = palette[idx % len(palette)]
            linewidth = 2.5 if "BiLSTM" in name or "Transformer" in name else 1.8
            plt.plot(rec, prec, label=f"{name} (AUPRC = {pr_score:.3f})", color=color, linewidth=linewidth)

        plt.axhline(baseline_rate, color="k", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({baseline_rate:.2f})")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
        plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
        plt.title("Diagnostic Delay Prediction: Precision-Recall Curve Comparison", fontsize=13, fontweight="bold", pad=12)
        plt.legend(loc="lower left", frameon=True, fontsize=9.5)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[Plotter] Saved PR comparison figure to {out_path}")

    def plot_benchmark_bars(
        self,
        benchmark_df: pd.DataFrame,
        filename: str = "model_benchmark_comparison.png"
    ):
        """Plots comparison bar chart across F1, ROC-AUC, and PR-AUC."""
        plt.figure(figsize=(10, 5.5), dpi=300)
        
        melted = benchmark_df.melt(
            id_vars=["Model"],
            value_vars=["F1_Score", "ROC_AUC", "PR_AUC"],
            var_name="Metric",
            value_name="Score"
        )

        ax = sns.barplot(
            data=melted,
            x="Model",
            y="Score",
            hue="Metric",
            palette="Set2"
        )
        plt.title("Model Performance Benchmark Across Evaluation Metrics", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
        plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
        plt.ylim(0, 1.08)
        plt.legend(frameon=True, loc="lower right")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[Plotter] Saved benchmark bar figure to {out_path}")
