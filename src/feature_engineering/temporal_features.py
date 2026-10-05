"""Temporal Dynamics Feature Extraction Module.

Extracts longitudinal inter-post interval statistics, posting velocity,
burstiness index, and timeline acceleration metrics.
"""

from typing import Dict, List
import numpy as np


class TemporalFeatureExtractor:
    """Extracts temporal features from timestamps and interval series."""

    def extract_features(
        self,
        timestamps_sec: List[float],
        delta_days: List[float]
    ) -> Dict[str, float]:
        """Extracts temporal features for a single patient timeline.
        
        Args:
            timestamps_sec: List of post timestamps in seconds.
            delta_days: List of days elapsed between consecutive posts.
            
        Returns:
            Dictionary of numerical temporal features.
        """
        if len(timestamps_sec) <= 1:
            return {
                "total_timeline_span_days": 0.0,
                "avg_gap_days": 0.0,
                "std_gap_days": 0.0,
                "max_gap_days": 0.0,
                "min_gap_days": 0.0,
                "posting_frequency_monthly": 1.0,
                "burstiness_index": 0.0,
                "gap_acceleration_slope": 0.0
            }

        span_days = max(1.0, (timestamps_sec[-1] - timestamps_sec[0]) / 86400.0)
        num_posts = len(timestamps_sec)
        
        gaps = np.array(delta_days[1:]) if len(delta_days) > 1 else np.array([0.0])
        avg_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        max_gap = float(np.max(gaps))
        min_gap = float(np.min(gaps))
        
        # Monthly frequency: posts per 30 days
        monthly_freq = (num_posts / span_days) * 30.0
        
        # Burstiness metric B = (std - mean) / (std + mean)
        denom = std_gap + avg_gap
        burstiness = (std_gap - avg_gap) / denom if denom > 1e-5 else 0.0
        
        # Gap trend (slope): positive slope = gaps getting longer; negative = gaps getting shorter (escalation)
        if len(gaps) >= 2:
            x = np.arange(len(gaps))
            slope, _ = np.polyfit(x, gaps, deg=1)
            gap_slope = float(slope)
        else:
            gap_slope = 0.0

        return {
            "total_timeline_span_days": round(span_days, 2),
            "avg_gap_days": round(avg_gap, 2),
            "std_gap_days": round(std_gap, 2),
            "max_gap_days": round(max_gap, 2),
            "min_gap_days": round(min_gap, 2),
            "posting_frequency_monthly": round(monthly_freq, 3),
            "burstiness_index": round(burstiness, 3),
            "gap_acceleration_slope": round(gap_slope, 3)
        }
