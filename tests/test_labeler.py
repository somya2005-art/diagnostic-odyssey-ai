"""Unit tests for DiagnosticDelayLabeler."""

import unittest
from src.preprocessing.labeler import DiagnosticDelayLabeler


class TestDiagnosticDelayLabeler(unittest.TestCase):

    def setUp(self):
        self.labeler = DiagnosticDelayLabeler(delay_threshold_years=1.0)

    def test_extract_years(self):
        text = "After struggling with symptoms, it took me 3 years to get diagnosed with Lupus."
        duration, conf, snippet = self.labeler.extract_duration_from_text(text)
        
        self.assertEqual(duration, 3.0)
        self.assertGreater(conf, 0.7)
        self.assertIn("took me 3 years to get diagnosed", snippet)

    def test_extract_months(self):
        text = "I was officially diagnosed after 6 months of joint stiffness."
        duration, conf, snippet = self.labeler.extract_duration_from_text(text)
        
        self.assertEqual(duration, 0.5)
        self.assertGreater(conf, 0.7)

    def test_word_numbers(self):
        text = "It took four years to finally get a diagnosis."
        duration, conf, snippet = self.labeler.extract_duration_from_text(text)
        
        self.assertEqual(duration, 4.0)

    def test_label_thresholding(self):
        long_timeline = [
            "My joints hurt.",
            "Doctor said it's anxiety.",
            "Took me 4 years to finally get diagnosed."
        ]
        label, dur, conf, _ = self.labeler.label_patient_timeline(long_timeline)
        self.assertEqual(label, 1, "4 years should be classified as Long Delay (1).")
        self.assertEqual(dur, 4.0)

        short_timeline = [
            "My knees are swollen.",
            "Saw a rheumatologist.",
            "Finally diagnosed after 4 months."
        ]
        label, dur, conf, _ = self.labeler.label_patient_timeline(short_timeline)
        self.assertEqual(label, 0, "4 months should be classified as Short Delay (0).")
        self.assertAlmostEqual(dur, 4.0 / 12.0, places=2)


if __name__ == "__main__":
    unittest.main()
