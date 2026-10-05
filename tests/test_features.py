"""Unit tests for domain feature extractors."""

import unittest
from src.feature_engineering.lexicon_extractor import DomainLexiconExtractor
from src.feature_engineering.temporal_features import TemporalFeatureExtractor
from src.feature_engineering.sentiment_trajectory import SentimentTrajectoryExtractor


class TestFeatureExtractors(unittest.TestCase):

    def setUp(self):
        self.lex_ext = DomainLexiconExtractor()
        self.temp_ext = TemporalFeatureExtractor()
        self.sent_ext = SentimentTrajectoryExtractor()

    def test_dismissal_and_symptom_lexicon(self):
        text = "Doctor told me it's all in my head and refused to test. I have severe malar rash and joint pain."
        feats = self.lex_ext.extract_features_from_text(text)
        
        self.assertGreaterEqual(feats["dismissal_count"], 1.0)
        self.assertEqual(feats["has_dismissal"], 1.0)
        self.assertGreaterEqual(feats["sym_mucocutaneous_count"], 1.0)
        self.assertGreaterEqual(feats["sym_musculoskeletal_count"], 1.0)
        self.assertGreaterEqual(feats["total_symptoms_count"], 2.0)

    def test_temporal_features(self):
        timestamps = [1000.0, 1000.0 + 86400 * 10, 1000.0 + 86400 * 40]
        delta_days = [0.0, 10.0, 30.0]
        feats = self.temp_ext.extract_features(timestamps, delta_days)
        
        self.assertEqual(feats["total_timeline_span_days"], 40.0)
        self.assertEqual(feats["avg_gap_days"], 20.0)
        self.assertEqual(feats["max_gap_days"], 30.0)
        self.assertGreater(feats["gap_acceleration_slope"], 0.0)

    def test_sentiment_trajectory(self):
        timeline = [
            "Wondering if anyone has advice on what tests to ask for?",
            "Doctor dismissed me. Feeling hopeless and exhausted and crying."
        ]
        feats = self.sent_ext.extract_trajectory_features(timeline)
        
        self.assertGreater(feats["mean_frustration"], 0.0)
        self.assertGreater(feats["frustration_trajectory_slope"], 0.0, "Frustration should increase over time.")


if __name__ == "__main__":
    unittest.main()
