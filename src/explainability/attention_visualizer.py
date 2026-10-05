"""Attention Trajectory Visualizer for Longitudinal Diagnostic Odyssey.

Visualizes attention weights assigned by the BiLSTM-Attention model across
a patient's sequence of pre-diagnostic posts over elapsed calendar time.
"""

from typing import Dict, List, Optional
import numpy as np


class TimelineAttentionVisualizer:
    """Visualizes model attention over longitudinal patient post sequences."""

    @staticmethod
    def plot_patient_attention(
        patient_id: str,
        posts_text: List[str],
        cumulative_days: List[float],
        attention_weights: np.ndarray,
        predicted_prob: float,
        true_label: Optional[int] = None,
        output_path: Optional[str] = None
    ):
        """Generates a publication-grade timeline attention plot for an individual patient."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("[TimelineAttentionVisualizer] matplotlib/seaborn not installed, skipping plot.")
            return
        # Trim attention weights to actual sequence length
        n_posts = len(posts_text)
        valid_attn = attention_weights[-n_posts:] if len(attention_weights) >= n_posts else attention_weights
        # Normalize to sum to 1.0
        if np.sum(valid_attn) > 0:
            valid_attn = valid_attn / np.sum(valid_attn)

        # Generate short post previews
        post_labels = []
        for i, text in enumerate(posts_text):
            day_str = f"Day {int(cumulative_days[i])}" if i < len(cumulative_days) else f"Post {i+1}"
            snippet = (text[:45] + "...") if len(text) > 45 else text
            post_labels.append(f"[{day_str}]\n{snippet}")

        plt.figure(figsize=(10, 5), dpi=300)
        sns.set_theme(style="whitegrid")

        # Color bar based on delay prediction
        colors = ["#e74c3c" if p == np.argmax(valid_attn) else "#3498db" for p in range(len(valid_attn))]

        x_indices = np.arange(n_posts)
        bars = plt.bar(x_indices, valid_attn, color=colors, alpha=0.85, width=0.55, edgecolor="black", linewidth=0.8)

        # Annotate peak attention turning point
        peak_idx = int(np.argmax(valid_attn))
        plt.annotate(
            "★ Peak Diagnostic Milestone",
            xy=(peak_idx, valid_attn[peak_idx]),
            xytext=(peak_idx, valid_attn[peak_idx] + 0.07),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#c0392b",
            arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=7)
        )

        plt.xticks(x_indices, post_labels, rotation=15, ha="right", fontsize=9)
        plt.ylabel("Attention Weight (α_t)", fontsize=11, fontweight="bold")
        plt.ylim(0, max(valid_attn) * 1.35)

        label_str = f"True: {'Long Delay (>1 yr)' if true_label == 1 else 'Short Delay (<=1 yr)'}" if true_label is not None else ""
        title_str = (
            f"Longitudinal Model Attention over Timeline: {patient_id}\n"
            f"Predicted Delay Risk: {predicted_prob:.1%} | {label_str}"
        )
        plt.title(title_str, fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"[AttentionVisualizer] Saved patient attention plot to {output_path}")
        else:
            plt.show()
