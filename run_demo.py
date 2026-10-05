"""Execution Script for Diagnostic Delay NLP Research Pipeline.

Usage:
    python run_demo.py [--num_patients 150] [--config configs/config.yaml]
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import DiagnosticDelayPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Predicting Diagnostic Delay Signals from Patient Narratives (Reddit/Health Forums)"
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=150,
        help="Number of synthetic patient trajectories to simulate if no raw data provided (default: 150)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to project configuration file"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  PREDICTING DIAGNOSTIC DELAY SIGNALS FROM PATIENT-AUTHORED SOCIAL NARRATIVES")
    print("=" * 80)

    pipeline = DiagnosticDelayPipeline(config_path=args.config)

    # 1. Ingestion, Anonymization & Preprocessing
    valid_timelines, df_merged = pipeline.prepare_dataset(num_synthetic_patients=args.num_patients)

    # 2. Multi-Modal Feature Extraction
    X_tabular, y, feature_names, seq_tensors, seq_masks = pipeline.extract_features(valid_timelines)

    # 3. Model Training, Benchmarking, Explainability, & Visualization
    results = pipeline.run_benchmark_and_evaluation(
        X_tabular=X_tabular,
        y=y,
        sequence_tensors=seq_tensors,
        sequence_masks=seq_masks,
        feature_names=feature_names,
        timelines=valid_timelines
    )

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETE")
    print("  - Figures generated in: reports/figures/")
    print("  - Case studies generated in: reports/case_studies/")
    print("=" * 80)


if __name__ == "__main__":
    main()
