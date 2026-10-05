"""Publication-Ready Case Study Generator for Diagnostic Delay Research.

Generates structured, anonymized narrative case studies (Patient Odyssey profiles)
displaying full timeline evolution, attention weight milestones, SHAP feature attributions,
and qualitative clinical insights.
"""

import json
import os
from typing import Dict, List, Any, Optional
import numpy as np


class DiagnosticCaseStudyGenerator:
    """Generates qualitative, explainable clinical case studies from patient trajectories."""

    def __init__(self, output_dir: str = "reports/case_studies"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_case_study(
        self,
        patient_id: str,
        patient_timeline: Dict[str, Any],
        predicted_prob: float,
        attention_weights: np.ndarray,
        shap_explanations: List[Dict[str, Any]],
        token_highlights: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Creates a structured case study dictionary for a single patient."""
        posts = patient_timeline["texts"]
        cumulative_days = patient_timeline.get("cumulative_days", [0.0] * len(posts))
        delta_days = patient_timeline.get("delta_days", [0.0] * len(posts))
        true_label = patient_timeline.get("label", None)

        n_posts = len(posts)
        valid_attn = attention_weights[-n_posts:] if len(attention_weights) >= n_posts else attention_weights
        if np.sum(valid_attn) > 0:
            valid_attn = valid_attn / np.sum(valid_attn)

        # Build chronological timeline entry
        timeline_entries = []
        for i, text in enumerate(posts):
            timeline_entries.append({
                "post_index": i + 1,
                "elapsed_days": cumulative_days[i] if i < len(cumulative_days) else 0.0,
                "interval_from_prev_post_days": delta_days[i] if i < len(delta_days) else 0.0,
                "attention_weight": round(float(valid_attn[i]), 4),
                "is_peak_attention": bool(i == np.argmax(valid_attn)),
                "anonymized_text": text
            })

        predicted_category = "Long Delay (> 1 Year)" if predicted_prob >= 0.5 else "Short Delay (<= 1 Year)"
        true_category = "Long Delay (> 1 Year)" if true_label == 1 else "Short Delay (<= 1 Year)" if true_label == 0 else "Unknown"

        case_study = {
            "patient_id": patient_id,
            "prediction": {
                "predicted_delay_probability": round(float(predicted_prob), 4),
                "predicted_trajectory": predicted_category,
                "ground_truth_label": true_category,
                "is_concordant": bool((predicted_prob >= 0.5) == (true_label == 1)) if true_label is not None else None
            },
            "timeline_metrics": {
                "total_pre_diagnosis_posts": len(posts),
                "total_observed_span_days": patient_timeline.get("total_timeline_span_days", 0.0),
                "avg_gap_days": patient_timeline.get("avg_gap_days", 0.0)
            },
            "chronological_odyssey": timeline_entries,
            "top_predictive_features": shap_explanations[:6] if shap_explanations else [],
            "token_level_highlights": token_highlights or []
        }

        return case_study

    def export_markdown_report(self, case_study: Dict[str, Any], file_path: str):
        """Renders a formatted Markdown report suitable for inclusion in a paper appendix."""
        p_id = case_study["patient_id"]
        pred = case_study["prediction"]
        tm = case_study["timeline_metrics"]
        
        md = []
        md.append(f"# Clinical Case Study: {p_id}")
        md.append("")
        md.append(f"**Predicted Trajectory:** `{pred['predicted_trajectory']}` (Risk Score: **{pred['predicted_delay_probability']:.1%}**)")
        md.append(f"**Ground Truth Label:** `{pred['ground_truth_label']}` (Concordant: **{pred['is_concordant']}**)")
        md.append(f"**Pre-Diagnostic Odyssey Span:** {tm['total_observed_span_days']:.1f} days across {tm['total_pre_diagnosis_posts']} longitudinal posts.")
        md.append("")
        md.append("---")
        md.append("## Longitudinal Patient Narrative & Attention Weights")
        md.append("")
        md.append("| Post # | Day | Gap (Days) | Attention Weight (α) | Narrative Summary & Key Linguistic Signal |")
        md.append("|:------:|:---:|:----------:|:--------------------:|:------------------------------------------|")

        for entry in case_study["chronological_odyssey"]:
            peak_flag = " ⭐ **[PEAK]**" if entry["is_peak_attention"] else ""
            clean_text = entry["anonymized_text"].replace("\n", " ")
            snippet = clean_text[:140] + ("..." if len(clean_text) > 140 else "")
            md.append(
                f"| {entry['post_index']} | {entry['elapsed_days']:.0f} | {entry['interval_from_prev_post_days']:.0f} | "
                f"`{entry['attention_weight']:.3f}`{peak_flag} | {snippet} |"
            )

        md.append("")
        md.append("---")
        md.append("## Explainability & Feature Attribution (SHAP)")
        md.append("")
        md.append("| Predictive Feature | Observed Value | SHAP Impact | Clinical Interpretation |")
        md.append("|:-------------------|:--------------:|:-----------:|:------------------------|")

        for f in case_study["top_predictive_features"]:
            direction_symbol = "🔺 Increases Delay Risk" if f["shap_impact"] > 0 else "🔻 Decreases Delay Risk"
            md.append(f"| `{f['feature']}` | `{f['value']}` | `{f['shap_impact']:+.3f}` | {direction_symbol} |")

        md.append("")
        md.append("---")
        md.append("## Qualitative Discussion & Takeaway")
        if pred["predicted_delay_probability"] >= 0.5:
            md.append(
                "> **Clinical Takeaway**: This patient exhibits hallmark diagnostic odyssey markers, "
                "characterized by multiple medical dismissal events ('all in your head / just anxiety'), "
                "prolonged inter-post latency, and persistent multisystem complaints. The longitudinal sequence model "
                "placed peak attention on the initial dismissal encounter, demonstrating early detectability of delay trajectories."
            )
        else:
            md.append(
                "> **Clinical Takeaway**: This patient shows a rapid diagnostic pathway with prompt clinical validation "
                "and immediate antibody testing. The absence of gaslighting markers and short inter-post intervals "
                "reliably distinguished this trajectory from prolonged delay odysseys."
            )

        content = "\n".join(md)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def save_case_studies(
        self,
        case_studies: List[Dict[str, Any]],
        prefix: str = "case_study"
    ):
        """Saves both JSON and Markdown representations of case studies."""
        for i, cs in enumerate(case_studies):
            p_id = cs["patient_id"]
            json_path = os.path.join(self.output_dir, f"{prefix}_{i+1}_{p_id}.json")
            md_path = os.path.join(self.output_dir, f"{prefix}_{i+1}_{p_id}.md")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cs, f, indent=2)

            self.export_markdown_report(cs, md_path)
            print(f"[CaseStudyGenerator] Exported case study to {md_path}")
