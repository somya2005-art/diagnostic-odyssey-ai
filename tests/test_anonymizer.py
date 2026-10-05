"""Unit tests for PatientAnonymizer."""

import unittest
import pandas as pd
from src.preprocessing.anonymizer import PatientAnonymizer


class TestPatientAnonymizer(unittest.TestCase):

    def setUp(self):
        self.anonymizer = PatientAnonymizer(salt="TestSalt123")

    def test_hash_author_id(self):
        raw_author = "ChronicIllnessWarrior99"
        p_id_1 = self.anonymizer.hash_author_id(raw_author)
        p_id_2 = self.anonymizer.hash_author_id(raw_author)
        
        self.assertTrue(p_id_1.startswith("PATIENT_"))
        self.assertEqual(p_id_1, p_id_2, "Hashing must be deterministic for identical handles.")
        
        different_author = "LupusSpoonie12"
        p_id_diff = self.anonymizer.hash_author_id(different_author)
        self.assertNotEqual(p_id_1, p_id_diff, "Different authors must have distinct hashes.")

    def test_scrub_text_pii(self):
        text = (
            "Hi /u/HealthHelper, my doctor Dr. Alice Smith at Mayo Clinic gave me advice. "
            "Email me at user@test.com or call (555) 123-4567 or visit https://myhealth.com"
        )
        scrubbed = self.anonymizer.scrub_text(text)
        
        self.assertNotIn("/u/HealthHelper", scrubbed)
        self.assertIn("[REDACTED_USER]", scrubbed)
        self.assertNotIn("Dr. Alice Smith", scrubbed)
        self.assertIn("[REDACTED_PHYSICIAN]", scrubbed)
        self.assertNotIn("Mayo Clinic", scrubbed)
        self.assertIn("[REDACTED_CLINIC]", scrubbed)
        self.assertNotIn("user@test.com", scrubbed)
        self.assertIn("[REDACTED_EMAIL]", scrubbed)
        self.assertNotIn("555", scrubbed)
        self.assertIn("[REDACTED_PHONE]", scrubbed)
        self.assertNotIn("https://myhealth.com", scrubbed)
        self.assertIn("[REDACTED_URL]", scrubbed)

    def test_anonymize_dataframe(self):
        df = pd.DataFrame({
            "author": ["Alice", "Bob"],
            "text": ["Contact u/AliceDoc", "Check https://test.org"]
        })
        anon_df = self.anonymizer.anonymize_dataframe(df)
        
        self.assertNotIn("author", anon_df.columns)
        self.assertIn("patient_id", anon_df.columns)
        self.assertTrue(anon_df["text"].str.contains(r"\[REDACTED_").all())


if __name__ == "__main__":
    unittest.main()
