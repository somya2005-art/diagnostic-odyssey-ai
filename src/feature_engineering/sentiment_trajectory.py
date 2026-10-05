"""Sentiment & Emotional Trajectory Feature Extraction Module.

Quantifies affective dynamics across patient longitudinal timelines, including
frustration trajectory slope, validation-seeking, and emotional despair markers.
"""

import re
from typing import Dict, List
import numpy as np


class SentimentTrajectoryExtractor:
    """Extracts affective tone and longitudinal emotional trajectory features."""

    def __init__(self):
        # Emotional lexicons tailored to chronic illness odysseys
        self.frustration_words = [
            r"\b(?:hopeless|crying|gaslit|gaslighting|ignored|exhausted|frustrated|depressed|scared|terrified|losing\s+my\s+mind|crazy|give\s+up|breaking\s+point|suffering|awful|horrible|nightmare|nobody\s+cares)\b"
        ]
        self.inquisitive_words = [
            r"\b(?:advice|recommendations?|anyone\s+else|similar|what\s+tests|could\s+this\s+be|help\s+me\s+understand|experiences?|wondering|questions?)\b"
        ]
        self.positive_words = [
            r"\b(?:hopeful|relieved|grateful|thankful|supportive|kind|listened|helped|improving|better)\b"
        ]

        self.compiled_frust = [re.compile(p, re.IGNORECASE) for p in self.frustration_words]
        self.compiled_inq = [re.compile(p, re.IGNORECASE) for p in self.inquisitive_words]
        self.compiled_pos = [re.compile(p, re.IGNORECASE) for p in self.positive_words]

    def _score_post_affect(self, text: str) -> Dict[str, float]:
        """Calculates emotional intensity scores for an individual post."""
        words = text.split()
        n_words = max(1, len(words))
        
        frust_hits = sum(len(p.findall(text)) for p in self.compiled_frust)
        inq_hits = sum(len(p.findall(text)) for p in self.compiled_inq)
        pos_hits = sum(len(p.findall(text)) for p in self.compiled_pos)

        frust_score = (frust_hits / (n_words / 100.0))
        pos_score = (pos_hits / (n_words / 100.0))
        net_polarity = pos_score - frust_score

        return {
            "frustration_score": float(frust_score),
            "inquisitive_score": float(inq_hits),
            "positive_score": float(pos_score),
            "net_polarity": float(net_polarity)
        }

    def extract_trajectory_features(self, timeline_texts: List[str]) -> Dict[str, float]:
        """Calculates emotional trajectory slopes and summary stats over a patient's post sequence."""
        if not timeline_texts:
            return {
                "mean_frustration": 0.0,
                "max_frustration": 0.0,
                "frustration_trajectory_slope": 0.0,
                "mean_inquisitive": 0.0,
                "mean_net_polarity": 0.0,
                "polarity_drift_slope": 0.0
            }

        post_scores = [self._score_post_affect(t) for t in timeline_texts]
        frust_series = [s["frustration_score"] for s in post_scores]
        polarity_series = [s["net_polarity"] for s in post_scores]
        inq_series = [s["inquisitive_score"] for s in post_scores]

        mean_frust = float(np.mean(frust_series))
        max_frust = float(np.max(frust_series))
        mean_inq = float(np.mean(inq_series))
        mean_polarity = float(np.mean(polarity_series))

        # Compute longitudinal slopes
        if len(frust_series) >= 2:
            x = np.arange(len(frust_series))
            f_slope, _ = np.polyfit(x, frust_series, deg=1)
            p_slope, _ = np.polyfit(x, polarity_series, deg=1)
            frust_slope = float(f_slope)
            polarity_slope = float(p_slope)
        else:
            frust_slope = 0.0
            polarity_slope = 0.0

        return {
            "mean_frustration": round(mean_frust, 3),
            "max_frustration": round(max_frust, 3),
            "frustration_trajectory_slope": round(frust_slope, 3),
            "mean_inquisitive": round(mean_inq, 3),
            "mean_net_polarity": round(mean_polarity, 3),
            "polarity_drift_slope": round(polarity_slope, 3)
        }
