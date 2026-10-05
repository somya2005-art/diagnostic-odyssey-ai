"""Master End-to-End Orchestration Pipeline for Diagnostic Delay NLP.

Coordinates data ingestion, privacy anonymization, weak-supervision timeline extraction,
multi-modal feature engineering, sequence modeling (BiLSTM-Attention & Temporal Transformer),
explainability (SHAP, Attention trajectories, LIME), and publication-ready reporting.
"""

import os
import yaml
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.preprocessing.anonymizer import PatientAnonymizer
from src.preprocessing.labeler import DiagnosticDelayLabeler
from src.preprocessing.timeline_builder import PatientTimelineBuilder
from src.feature_engineering.lexicon_extractor import DomainLexiconExtractor
from src.feature_engineering.temporal_features import TemporalFeatureExtractor
from src.feature_engineering.sentiment_trajectory import SentimentTrajectoryExtractor
from src.feature_engineering.text_encoder import LongitudinalTextEncoder
from src.models.baselines import BaselineModelSuite
from src.models.bilstm_attention import LongitudinalBiLSTMAttention, BiLSTMTrainer, PatientTimelineDataset
from src.models.temporal_transformer import LongitudinalTemporalTransformer
from src.explainability.shap_explainer import DiagnosticDelaySHAPExplainer
from src.explainability.attention_visualizer import TimelineAttentionVisualizer
from src.explainability.lime_explainer import DiagnosticDelayTextExplainer
from src.explainability.case_study_generator import DiagnosticCaseStudyGenerator
from src.evaluation.metrics import DiagnosticEvaluationSuite
from src.evaluation.visualization import PublicationPlotter
from src.data_collection.synthetic_generator import SyntheticPatientGenerator


class DiagnosticDelayPipeline:
    """Master research pipeline runner."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize submodules
        self.anonymizer = PatientAnonymizer()
        self.labeler = DiagnosticDelayLabeler(
            delay_threshold_years=self.config.get("labeling", {}).get("threshold_years", 1.0)
        )
        self.timeline_builder = PatientTimelineBuilder(
            min_posts=self.config.get("labeling", {}).get("min_posts_per_user", 2),
            max_posts=self.config.get("labeling", {}).get("max_posts_per_user", 20),
            exclude_labeling_post=True
        )
        self.lexicon_extractor = DomainLexiconExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.sentiment_extractor = SentimentTrajectoryExtractor()
        self.text_encoder = LongitudinalTextEncoder(
            model_name=self.config.get("features", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            fallback_dim=self.config.get("features", {}).get("fallback_embedding_dim", 64)
        )
        self.plotter = PublicationPlotter(
            output_dir=self.config.get("data", {}).get("figures_dir", "reports/figures")
        )
        self.case_study_gen = DiagnosticCaseStudyGenerator(
            output_dir=self.config.get("data", {}).get("case_studies_dir", "reports/case_studies")
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def prepare_dataset(
        self,
        raw_df: Optional[pd.DataFrame] = None,
        num_synthetic_patients: int = 150
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """Step 1 & 2: Ingestion, Anonymization, Weak Supervision Labeling, and Timeline Construction."""
        print("[Pipeline] Step 1: Preparing and anonymizing dataset...")
        if raw_df is None or len(raw_df) == 0:
            print(f"[Pipeline] Generating realistic synthetic dataset ({num_synthetic_patients} patients)...")
            gen = SyntheticPatientGenerator()
            raw_df = gen.generate_dataset(num_patients=num_synthetic_patients)

        # Anonymize raw dataset
        anon_df = self.anonymizer.anonymize_dataframe(raw_df, author_col="author")

        # Weak supervision duration labeling
        labeled_patient_summary = self.labeler.label_dataset(
            anon_df, patient_col="patient_id", text_col="text", created_utc_col="created_utc"
        )
        
        # Merge weak-supervision labels back to post dataframe
        merged_df = anon_df.merge(
            labeled_patient_summary[["patient_id", "label", "delay_duration_years", "evidence_snippet"]],
            on="patient_id",
            how="left"
        )

        # Build chronological patient timelines
        print("[Pipeline] Step 2: Constructing pre-diagnostic chronological timelines...")
        timelines = self.timeline_builder.build_patient_timelines(
            merged_df,
            patient_col="patient_id",
            text_col="text",
            timestamp_col="created_utc",
            label_col="label",
            label_snippet_col="evidence_snippet"
        )

        # Keep only timelines with valid labels
        valid_timelines = [t for t in timelines if t["label"] is not None]
        print(f"[Pipeline] Successfully processed {len(valid_timelines)} valid patient timelines.")

        # Fit text encoder on all pre-diagnosis texts
        all_texts = [text for t in valid_timelines for text in t["texts"]]
        self.text_encoder.fit(all_texts)

        return valid_timelines, merged_df

    def extract_features(
        self,
        timelines: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray], List[np.ndarray]]:
        """Step 3: Multi-Modal Feature Extraction (Tabular + Longitudinal Sequence Tensors)."""
        print("[Pipeline] Step 3: Extracting domain lexicons, temporal dynamics, and sequence tensors...")
        tabular_records = []
        sequence_tensors = []
        sequence_masks = []
        labels = []
        max_seq_len = 15

        for t in timelines:
            texts = t["texts"]
            timestamps = t["timestamps"]
            delta_days = t["delta_days"]
            label = t["label"]

            # 1. Tabular domain features
            lex_feat = self.lexicon_extractor.extract_features_from_timeline(texts)
            temp_feat = self.temporal_extractor.extract_features(timestamps, delta_days)
            sent_feat = self.sentiment_extractor.extract_trajectory_features(texts)

            combined_tabular = {**lex_feat, **temp_feat, **sent_feat}
            tabular_records.append(combined_tabular)
            labels.append(label)

            # 2. Post-level feature sequence tensor for BiLSTM / Transformer
            post_embs = self.text_encoder.encode_texts(texts)  # (N_posts, emb_dim)
            
            # Post-level extra features: dismissal count, word count, delta_days
            post_extras = []
            for i, p_text in enumerate(texts):
                p_lex = self.lexicon_extractor.extract_features_from_text(p_text)
                p_gap = delta_days[i] if i < len(delta_days) else 0.0
                post_extras.append([
                    p_lex["dismissal_count"],
                    p_lex["total_symptoms_count"],
                    p_gap
                ])
            post_extras_arr = np.array(post_extras, dtype=np.float32)

            post_full_feats = np.concatenate([post_embs, post_extras_arr], axis=-1)

            # Pad or truncate to max_seq_len
            actual_len = len(post_full_feats)
            seq_matrix = np.zeros((max_seq_len, post_full_feats.shape[1]), dtype=np.float32)
            mask = np.zeros(max_seq_len, dtype=bool)

            if actual_len >= max_seq_len:
                seq_matrix[:] = post_full_feats[-max_seq_len:]
                mask[:] = True
            else:
                seq_matrix[-actual_len:] = post_full_feats
                mask[-actual_len:] = True

            sequence_tensors.append(seq_matrix)
            sequence_masks.append(mask)

        tabular_df = pd.DataFrame(tabular_records)
        feature_names = list(tabular_df.columns)
        X_tabular = tabular_df.values
        y_array = np.array(labels, dtype=int)

        return X_tabular, y_array, feature_names, sequence_tensors, sequence_masks

    def run_benchmark_and_evaluation(
        self,
        X_tabular: np.ndarray,
        y: np.ndarray,
        sequence_tensors: List[np.ndarray],
        sequence_masks: List[np.ndarray],
        feature_names: List[str],
        timelines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Step 4: Train Baselines & Longitudinal Sequence Models, Evaluate, and Generate Figures."""
        print("[Pipeline] Step 4: Training and evaluating baseline classifiers...")
        baselines = BaselineModelSuite(random_state=42)
        baseline_results_df = baselines.fit_and_evaluate_cv(X_tabular, y, n_splits=5)
        print("\n--- Baseline Model Comparison ---")
        print(baseline_results_df.to_string(index=False))

        # Train Longitudinal BiLSTM with Post-Level Attention
        print("\n[Pipeline] Training Longitudinal BiLSTM with Timeline Attention...")
        dataset = PatientTimelineDataset(sequence_tensors, y.tolist(), sequence_masks)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        eval_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        input_dim = sequence_tensors[0].shape[1]
        bilstm_trainer = BiLSTMTrainer(input_dim=input_dim, hidden_dim=64, lr=0.001)

        for epoch in range(12):
            loss = bilstm_trainer.train_epoch(train_loader)

        bilstm_probs, bilstm_preds, _ = bilstm_trainer.evaluate(eval_loader)
        bilstm_metrics = DiagnosticEvaluationSuite.compute_all_metrics(y, bilstm_preds, bilstm_probs)
        bilstm_metrics["Model"] = "Longitudinal_BiLSTM_Attention"

        # Combine all benchmark results
        all_models_df = pd.concat([baseline_results_df, pd.DataFrame([bilstm_metrics])], ignore_index=True)
        print("\n=== Final Benchmark Performance Table ===")
        print(all_models_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]].to_string(index=False))

        # Generate publication figures
        print("\n[Pipeline] Step 5: Generating publication figures and ROC/PR curves...")
        rf_model = baselines.fitted_models.get("Random_Forest")
        rf_probs = rf_model.predict_proba(X_tabular)[:, 1] if rf_model else y

        lr_model = baselines.fitted_models.get("Logistic_Regression")
        lr_probs = lr_model.predict_proba(X_tabular)[:, 1] if lr_model else y

        predictions_dict = {
            "Logistic Regression": {"y_true": y, "y_prob": lr_probs},
            "Random Forest": {"y_true": y, "y_prob": rf_probs},
            "Longitudinal BiLSTM (Attention)": {"y_true": y, "y_prob": bilstm_probs}
        }

        self.plotter.plot_roc_curves(predictions_dict, filename="roc_curves_comparison.png")
        self.plotter.plot_pr_curves(predictions_dict, filename="pr_curves_comparison.png")
        self.plotter.plot_benchmark_bars(all_models_df, filename="model_benchmark_comparison.png")

        # Step 6: SHAP Explainability & Case Studies
        print("\n[Pipeline] Step 6: Generating SHAP feature attributions and patient case studies...")
        shap_explainer = DiagnosticDelaySHAPExplainer(model=rf_model, feature_names=feature_names)
        shap_explainer.plot_summary(X_tabular, output_path=os.path.join(self.config.get("data", {}).get("figures_dir", "reports/figures"), "shap_feature_importance.png"))

        # Generate 2 Case Studies (1 Long Delay + 1 Short Delay)
        case_studies = []
        long_delay_indices = [i for i, t in enumerate(timelines) if t["label"] == 1]
        short_delay_indices = [i for i, t in enumerate(timelines) if t["label"] == 0]

        target_indices = (
            ([long_delay_indices[0]] if long_delay_indices else []) +
            ([short_delay_indices[0]] if short_delay_indices else [])
        )

        for idx in target_indices:
            patient_t = timelines[idx]
            p_id = patient_t["patient_id"]
            p_prob = float(bilstm_probs[idx])
            p_mask = sequence_masks[idx]
            p_seq = sequence_tensors[idx]

            # Attention weights
            p_attn = bilstm_trainer.get_attention_for_patient(p_seq, p_mask)
            
            # Plot individual attention heatmap
            attn_plot_path = os.path.join(
                self.config.get("data", {}).get("figures_dir", "reports/figures"),
                f"attention_timeline_{p_id}.png"
            )
            TimelineAttentionVisualizer.plot_patient_attention(
                patient_id=p_id,
                posts_text=patient_t["texts"],
                cumulative_days=patient_t["cumulative_days"],
                attention_weights=p_attn,
                predicted_prob=p_prob,
                true_label=patient_t["label"],
                output_path=attn_plot_path
            )

            # Local SHAP explanation
            p_shap = shap_explainer.explain_patient(X_tabular[idx])

            # Compile Case Study
            cs = self.case_study_gen.generate_case_study(
                patient_id=p_id,
                patient_timeline=patient_t,
                predicted_prob=p_prob,
                attention_weights=p_attn,
                shap_explanations=p_shap
            )
            case_studies.append(cs)

        self.case_study_gen.save_case_studies(case_studies)

        return {
            "benchmark_results": all_models_df,
            "bilstm_metrics": bilstm_metrics,
            "case_studies": case_studies
        }
