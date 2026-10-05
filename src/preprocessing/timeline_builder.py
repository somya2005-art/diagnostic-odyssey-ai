"""Timeline Builder for Longitudinal Sequence Modeling.

Constructs chronological patient timelines, computes inter-post delta-times,
handles pre-diagnostic data-leakage truncation, and generates sequence tensors.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class PatientTimelineBuilder:
    """Builds structured longitudinal post sequences for each patient."""

    def __init__(
        self,
        min_posts: int = 2,
        max_posts: int = 20,
        exclude_labeling_post: bool = True
    ):
        """Initialize the timeline builder.
        
        Args:
            min_posts: Minimum number of historical posts required for inclusion.
            max_posts: Maximum length of sequence (truncated to most recent pre-diagnosis posts).
            exclude_labeling_post: If True, removes the final post containing the explicit diagnosis announcement
                                  to ensure pure pre-diagnosis predictive validity without leakage.
        """
        self.min_posts = min_posts
        self.max_posts = max_posts
        self.exclude_labeling_post = exclude_labeling_post

    def build_patient_timelines(
        self,
        df: pd.DataFrame,
        patient_col: str = "patient_id",
        text_col: str = "text",
        timestamp_col: str = "created_utc",
        label_col: Optional[str] = "label",
        label_snippet_col: Optional[str] = "evidence_snippet"
    ) -> List[Dict[str, Any]]:
        """Constructs ordered patient timelines with delta times and metadata.
        
        Args:
            df: DataFrame of individual posts.
            patient_col: Patient ID column.
            text_col: Text column.
            timestamp_col: Timestamp in UTC seconds or datetime.
            label_col: Ground truth / weak supervision label column.
            label_snippet_col: Evidence snippet to identify and exclude the diagnosis announcement.
            
        Returns:
            List of patient timeline dictionaries.
        """
        df_sorted = df.sort_values(by=[patient_col, timestamp_col]).copy()
        
        # Ensure timestamp is numeric (seconds)
        if pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_col]):
            df_sorted["timestamp_sec"] = df_sorted[timestamp_col].astype(int) / 10**9
        else:
            df_sorted["timestamp_sec"] = pd.to_numeric(df_sorted[timestamp_col], errors="coerce").fillna(0)

        timelines = []

        for patient_id, group in df_sorted.groupby(patient_col):
            posts_text = group[text_col].tolist()
            timestamps = group["timestamp_sec"].tolist()
            
            # Determine label if available in dataset
            label = None
            if label_col and label_col in group.columns:
                non_null_labels = group[label_col].dropna()
                if len(non_null_labels) > 0:
                    label = int(non_null_labels.iloc[0])

            # Optionally prune the explicit diagnosis post to avoid target leakage
            if self.exclude_labeling_post and len(posts_text) > self.min_posts:
                # Exclude the last post if it contains the diagnostic announcement
                posts_text = posts_text[:-1]
                timestamps = timestamps[:-1]

            if len(posts_text) < self.min_posts:
                continue

            # Truncate to maximum sequence length (keeping the most recent posts leading to diagnosis)
            if len(posts_text) > self.max_posts:
                posts_text = posts_text[-self.max_posts:]
                timestamps = timestamps[-self.max_posts:]

            # Compute delta times (in days)
            delta_days = [0.0]
            cumulative_days = [0.0]
            start_ts = timestamps[0]

            for i in range(1, len(timestamps)):
                diff_sec = max(0.0, timestamps[i] - timestamps[i - 1])
                diff_days = diff_sec / 86400.0
                delta_days.append(round(diff_days, 2))
                
                cum_days = (timestamps[i] - start_ts) / 86400.0
                cumulative_days.append(round(cum_days, 2))

            timeline_dict = {
                "patient_id": patient_id,
                "label": label,
                "num_posts": len(posts_text),
                "texts": posts_text,
                "timestamps": timestamps,
                "delta_days": delta_days,
                "cumulative_days": cumulative_days,
                "total_timeline_span_days": cumulative_days[-1] if len(cumulative_days) > 0 else 0.0,
                "avg_gap_days": float(np.mean(delta_days[1:])) if len(delta_days) > 1 else 0.0
            }
            timelines.append(timeline_dict)

        return timelines

    def timelines_to_dataframe(self, timelines: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts structured timelines into a summary DataFrame."""
        records = []
        for t in timelines:
            records.append({
                "patient_id": t["patient_id"],
                "label": t["label"],
                "num_posts": t["num_posts"],
                "total_timeline_span_days": t["total_timeline_span_days"],
                "avg_gap_days": t["avg_gap_days"],
                "aggregated_text": " [SEP] ".join(t["texts"])
            })
        return pd.DataFrame(records)
