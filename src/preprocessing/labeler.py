"""Weak-Supervision Diagnostic Delay Labeling Module.

Extracts self-reported diagnostic duration and labels patient timelines
into Short Delay (<= 1.0 yr) vs. Long Delay (> 1.0 yr) with confidence scoring.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DiagnosticDelayLabeler:
    """Extracts diagnostic timeline signals and labels patient-authored posts."""

    def __init__(self, delay_threshold_years: float = 1.0):
        """Initialize labeler with a duration threshold.
        
        Args:
            delay_threshold_years: Cutoff in years. Duration > threshold is classified as 1 (Long Delay).
        """
        self.threshold = delay_threshold_years
        
        # Word-to-number mapping for textual numbers
        self.num_words = {
            "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
            "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
            "eleven": 11.0, "twelve": 12.0, "fifteen": 15.0, "twenty": 20.0,
            "a": 1.0, "an": 1.0, "several": 3.0, "couple": 2.0, "few": 3.0,
            "half a": 0.5, "half": 0.5
        }
        
        # Regex patterns for diagnostic duration extraction
        num_pattern = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|a|an|several|couple|few|half a|half)"
        time_unit = r"(?:years?|yrs?|months?|mos?|weeks?|wks?|decades?)"
        
        self.patterns = [
            # e.g., "took me 3 years to get diagnosed", "took 18 months to finally get a diagnosis"
            re.compile(
                rf"(?:took|taken|waited|suffered for|struggled for|spent|took me|lasted)\s+({num_pattern})\s+({time_unit})\s+(?:to\s+(?:finally\s+)?(?:get|receive|reach|find)\s+(?:a\s+)?(?:diagnosis|diagnosed|answers?)|before\s+(?:being\s+)?(?:diagnosed|getting\s+diagnosed))",
                re.IGNORECASE
            ),
            # e.g., "finally diagnosed after 5 years", "officially diagnosed after 6 months"
            re.compile(
                rf"(?:finally|officially|eventually|got|received|was)\s+diagnosed\s+(?:after|following)\s+({num_pattern})\s+({time_unit})",
                re.IGNORECASE
            ),
            # e.g., "diagnostic journey took 4 years", "diagnostic delay of 2 years"
            re.compile(
                rf"(?:diagnostic\s+(?:journey|odyssey|process|delay)|getting\s+a\s+diagnosis)\s+(?:took|lasted|was)\s+({num_pattern})\s+({time_unit})",
                re.IGNORECASE
            ),
            # e.g., "3 years from first symptom to diagnosis"
            re.compile(
                rf"({num_pattern})\s+({time_unit})\s+(?:from\s+(?:first\s+)?symptoms?\s+to\s+(?:a\s+)?diagnosis|between\s+first\s+symptom\s+and\s+diagnosis)",
                re.IGNORECASE
            ),
            # e.g., "undiagnosed for 7 years"
            re.compile(
                rf"(?:undiagnosed|without\s+a\s+diagnosis|searching\s+for\s+answers?)\s+for\s+({num_pattern})\s+({time_unit})",
                re.IGNORECASE
            ),
            # e.g., "it has been 4 years and still no diagnosis"
            re.compile(
                rf"(?:been|it\'s\s+been|suffering\s+for)\s+({num_pattern})\s+({time_unit})\s+(?:and\s+still\s+no\s+diagnosis|without\s+answers?|trying\s+to\s+get\s+diagnosed)",
                re.IGNORECASE
            )
        ]

    def _parse_num(self, num_str: str) -> Optional[float]:
        """Convert numeric or written number string into float."""
        num_str = num_str.strip().lower()
        if num_str in self.num_words:
            return self.num_words[num_str]
        try:
            return float(num_str)
        except ValueError:
            return None

    def _unit_to_years(self, unit_str: str) -> float:
        """Normalize unit to fractional years."""
        unit_str = unit_str.lower()
        if "year" in unit_str or "yr" in unit_str:
            return 1.0
        elif "decade" in unit_str:
            return 10.0
        elif "month" in unit_str or "mo" in unit_str:
            return 1.0 / 12.0
        elif "week" in unit_str or "wk" in unit_str:
            return 1.0 / 52.0
        return 1.0

    def extract_duration_from_text(self, text: str) -> Tuple[Optional[float], float, str]:
        """Extracts diagnostic duration in years, confidence score, and matching snippet.
        
        Args:
            text: Patient post or timeline text.
            
        Returns:
            Tuple of (duration_years, confidence_score [0.0-1.0], matching_pattern_text).
        """
        if not text or not isinstance(text, str):
            return None, 0.0, ""

        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                num_val = self._parse_num(match.group(1))
                unit_mult = self._unit_to_years(match.group(2))
                if num_val is not None and num_val > 0:
                    duration_years = round(num_val * unit_mult, 3)
                    # Heuristic confidence: explicit diagnosis phrases have higher confidence
                    matched_snippet = match.group(0)
                    confidence = 0.95 if "diagnos" in matched_snippet.lower() else 0.80
                    return duration_years, confidence, matched_snippet

        return None, 0.0, ""

    def label_patient_timeline(
        self,
        timeline_texts: List[str]
    ) -> Tuple[Optional[int], Optional[float], float, str]:
        """Labels a patient based on all posts in their timeline.
        
        Args:
            timeline_texts: List of posts by a single patient in chronological order.
            
        Returns:
            Tuple of (binary_label [0 or 1], duration_years, confidence, matched_snippet).
        """
        # Search backwards from the latest post (often where diagnosis is announced)
        for text in reversed(timeline_texts):
            duration_years, conf, snippet = self.extract_duration_from_text(text)
            if duration_years is not None:
                binary_label = 1 if duration_years > self.threshold else 0
                return binary_label, duration_years, conf, snippet

        return None, None, 0.0, ""

    def label_dataset(
        self,
        df: pd.DataFrame,
        patient_col: str = "patient_id",
        text_col: str = "text",
        created_utc_col: str = "created_utc"
    ) -> pd.DataFrame:
        """Applies weak supervision labeling to a DataFrame grouped by patient.
        
        Args:
            df: DataFrame containing patient posts.
            patient_col: Column with patient identifier.
            text_col: Column with post text.
            created_utc_col: Column with timestamp.
            
        Returns:
            DataFrame of labeled patients with summary statistics.
        """
        sorted_df = df.sort_values(by=[patient_col, created_utc_col])
        patient_records = []

        for patient_id, group in sorted_df.groupby(patient_col):
            texts = group[text_col].tolist()
            label, duration, conf, snippet = self.label_patient_timeline(texts)
            
            patient_records.append({
                "patient_id": patient_id,
                "num_posts": len(texts),
                "is_labeled": label is not None,
                "label": label,
                "delay_duration_years": duration,
                "label_confidence": conf,
                "evidence_snippet": snippet
            })

        return pd.DataFrame(patient_records)
