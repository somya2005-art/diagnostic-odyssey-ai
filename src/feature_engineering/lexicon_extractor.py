"""Domain-Specific Lexicon Feature Extraction Module.

Extracts counts and normalized frequencies for:
1. Medical Dismissal & Gaslighting Signals
2. Autoimmune & Systemic Symptom Clusters (Musculoskeletal, Systemic, Mucocutaneous, Neuro, GI)
3. Healthcare System Friction & Odyssey Markers (Doctor counts, Misdiagnosis flags, Specialist visits)
"""

import re
from typing import Dict, List, Any
import numpy as np
import pandas as pd


class DomainLexiconExtractor:
    """Extracts clinical, dismissal, and healthcare friction lexicon features from patient texts."""

    def __init__(self):
        # 1. Medical dismissal & physician gaslighting patterns
        self.dismissal_lexicon = [
            r"\b(?:in\s+my\s+head|all\s+in\s+your\s+head)\b",
            r"\b(?:just\s+(?:stress|anxiety|depression|in\s+your\s+head|tired|overthinking))\b",
            r"\b(?:labs?\s+(?:are|were)\s+normal|blood\s*work\s+(?:is|was)\s+normal|everything\s+came\s+back\s+normal)\b",
            r"\b(?:refused\s+to\s+(?:test|run|order|listen|refer))\b",
            r"\b(?:doctor\s+(?:dismissed|ignored|laughed|didn\'t\s+listen|did\s+not\s+believe|gaslit|gaslighting))\b",
            r"\b(?:too\s+young\s+to\s+(?:have|be\s+sick))\b",
            r"\b(?:hypochondriac|somatization|somatic|making\s+it\s+up|exaggerating)\b",
            r"\b(?:stop\s+googling|stay\s+off\s+google|waste\s+of\s+time)\b",
            r"\b(?:no\s+one\s+believes\s+me|nobody\s+takes\s+me\s+seriously)\b"
        ]

        # 2. Symptom clusters
        self.symptom_clusters = {
            "sym_musculoskeletal": [
                r"\b(?:joint\s+pain|swollen\s+joints|knuckles|stiffness|morning\s+stiffness|arthritis|muscle\s+pain|myalgia|tendonitis|wrist\s+pain)\b"
            ],
            "sym_systemic": [
                r"\b(?:chronic\s+fatigue|debilitating\s+fatigue|exhaustion|low\s+grade\s+fevers?|night\s+sweats|unexplained\s+fevers?|flu[- ]like|malaise)\b"
            ],
            "sym_mucocutaneous": [
                r"\b(?:malar\s+rash|butterfly\s+rash|photosensitivity|sun\s+rash|dry\s+eyes|dry\s+mouth|mouth\s+sores|mouth\s+ulcers|hair\s+loss|alopecia|raynaud\'?s?)\b"
            ],
            "sym_neurological": [
                r"\b(?:brain\s+fog|memory\s+loss|tingling|pins\s+and\s+needles|numbness|neuropathy|dizziness|vertigo|pots|dysautonomia|migraines?)\b"
            ],
            "sym_gastrointestinal": [
                r"\b(?:nausea|stomach\s+pain|abdominal\s+pain|bloating|ibs|reflux|gerd|digestive\s+issues)\b"
            ]
        }

        # 3. Healthcare friction & diagnostic odyssey markers
        self.friction_lexicon = {
            "specialist_referrals": [
                r"\b(?:rheumatologist|neurologist|endocrinologist|gastroenterologist|hematologist|immunologist|specialist)\b"
            ],
            "doctor_shopping_counts": [
                r"\b(?:saw\s+(?:\d+|two|three|four|five|multiple|several)\s+(?:doctors|specialists|physicians))\b",
                r"\b(?:second\s+opinion|third\s+opinion|fourth\s+doctor|fifth\s+doctor)\b",
                r"\b(?:switching\s+doctors|fired\s+my\s+doctor|new\s+doctor)\b"
            ],
            "misdiagnoses": [
                r"\b(?:misdiagnosed|diagnosed\s+with\s+(?:fibromyalgia|fibro|anxiety|depression|ibs|chronic\s+fatigue\s+syndrome|cfs|somatoform))\b"
            ],
            "diagnostic_tests": [
                r"\b(?:ana|anti[- ]?dsdna|ssa|ssb|esr|crp|sed\s+rate|rheumatoid\s+factor|rf|biopsy|mri|complement|c3|c4)\b"
            ]
        }

        # Precompile all regex patterns
        self.compiled_dismissal = [re.compile(p, re.IGNORECASE) for p in self.dismissal_lexicon]
        self.compiled_symptoms = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.symptom_clusters.items()
        }
        self.compiled_friction = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.friction_lexicon.items()
        }

    def extract_features_from_text(self, text: str) -> Dict[str, float]:
        """Extracts lexicon frequency features from a single text string."""
        if not text or not isinstance(text, str):
            text = ""
        
        words = text.split()
        word_count = max(1, len(words))
        features = {}

        # 1. Dismissal score
        dismissal_hits = sum(len(p.findall(text)) for p in self.compiled_dismissal)
        features["dismissal_count"] = float(dismissal_hits)
        features["dismissal_rate"] = float(dismissal_hits) / (word_count / 100.0)
        features["has_dismissal"] = 1.0 if dismissal_hits > 0 else 0.0

        # 2. Symptom cluster scores
        total_symptoms = 0
        for cluster_name, patterns in self.compiled_symptoms.items():
            cluster_hits = sum(len(p.findall(text)) for p in patterns)
            features[f"{cluster_name}_count"] = float(cluster_hits)
            features[f"{cluster_name}_rate"] = float(cluster_hits) / (word_count / 100.0)
            total_symptoms += cluster_hits

        features["total_symptoms_count"] = float(total_symptoms)
        features["symptom_multisystem_breadth"] = float(
            sum(1 for cluster in self.compiled_symptoms if features[f"{cluster}_count"] > 0)
        )

        # 3. Healthcare friction scores
        for friction_name, patterns in self.compiled_friction.items():
            friction_hits = sum(len(p.findall(text)) for p in patterns)
            features[f"{friction_name}_count"] = float(friction_hits)

        features["total_friction_count"] = float(
            sum(features[f"{k}_count"] for k in self.compiled_friction)
        )

        return features

    def extract_features_from_timeline(self, timeline_texts: List[str]) -> Dict[str, float]:
        """Extracts aggregated & longitudinal lexicon features across a patient's post sequence."""
        if not timeline_texts:
            return {k: 0.0 for k in self.get_feature_names()}

        post_features_list = [self.extract_features_from_text(t) for t in timeline_texts]
        aggregated_text = " ".join(timeline_texts)
        agg_features = self.extract_features_from_text(aggregated_text)

        # Compute trajectory: dismissal in early vs late posts
        early_posts = timeline_texts[:max(1, len(timeline_texts) // 2)]
        late_posts = timeline_texts[max(1, len(timeline_texts) // 2):]

        early_dismissals = sum(self.extract_features_from_text(t)["dismissal_count"] for t in early_posts)
        late_dismissals = sum(self.extract_features_from_text(t)["dismissal_count"] for t in late_posts)

        agg_features["early_dismissals"] = float(early_dismissals)
        agg_features["late_dismissals"] = float(late_dismissals)
        agg_features["dismissal_persistence_ratio"] = float(late_dismissals / (early_dismissals + 1e-5))
        agg_features["posts_with_dismissal_ratio"] = float(
            np.mean([pf["has_dismissal"] for pf in post_features_list])
        )

        return agg_features

    def get_feature_names(self) -> List[str]:
        """Returns the complete list of lexicon feature keys."""
        dummy = self.extract_features_from_timeline(["dummy post text"])
        return list(dummy.keys())
