"""Data Anonymization and Privacy Preservation Module
Compliant with the Association of Internet Researchers (AoIR) ethical guidelines.

Features:
- Salted SHA-256 pseudonymization for author handles
- PII scrubbing (usernames, emails, URLs, phone numbers, clinical names)
- Paraphrasing / text sanitization utilities
"""

import hashlib
import re
from typing import Dict, List, Optional, Union
import pandas as pd


class PatientAnonymizer:
    """Anonymizes patient identifiers and removes personally identifiable information (PII)."""

    def __init__(self, salt: str = "DiagnosticDelayStudy2026_SecSalt"):
        """Initialize the anonymizer with a cryptographic salt.
        
        Args:
            salt: Secret salt for hashing patient IDs to prevent rainbow table attacks.
        """
        self.salt = salt.encode("utf-8")
        
        # Regex patterns for PII scrubbing (ordered by precedence)
        self.patterns = [
            # 1. URLs (hyperlinks)
            (re.compile(r"https?://\S+|www\.\S+"), "[REDACTED_URL]"),
            # 2. Email addresses (must precede @mentions)
            (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[REDACTED_EMAIL]"),
            # 3. Reddit username mentions (u/username or /u/username)
            (re.compile(r"/?u/[A-Za-z0-9_-]+", re.IGNORECASE), "[REDACTED_USER]"),
            # 4. Generic @mentions
            (re.compile(r"@[A-Za-z0-9_-]+"), "[REDACTED_USER]"),
            # 5. Phone numbers (US/Intl patterns)
            (re.compile(r"(?:\(\d{3}\)|\b\d{3}\b)[-. ]?\d{3}[-. ]?\d{4}\b"), "[REDACTED_PHONE]"),
            # 6. Direct physician names (e.g., Dr. Smith, Dr Jones)
            (re.compile(r"\b(?:Dr\.|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"), "[REDACTED_PHYSICIAN]"),
            # 7. Specific clinic/hospital designations (e.g. Mayo Clinic, Johns Hopkins, Cleveland Clinic)
            (re.compile(r"\b(?:Mayo Clinic|Cleveland Clinic|Johns Hopkins|Mount Sinai|Kaiser|NHS Trust)\b", re.IGNORECASE), "[REDACTED_CLINIC]"),
        ]

    def hash_author_id(self, raw_author_id: str) -> str:
        """Deterministically hashes an author identifier using SHA-256 with salt.
        
        Args:
            raw_author_id: Raw string identifier (e.g. Reddit username).
            
        Returns:
            Truncated SHA-256 hash formatted as 'PATIENT_<hash>'.
        """
        if not raw_author_id or pd.isna(raw_author_id):
            return "PATIENT_UNKNOWN"
        raw_bytes = str(raw_author_id).strip().lower().encode("utf-8")
        hasher = hashlib.sha256(self.salt)
        hasher.update(raw_bytes)
        digest = hasher.hexdigest()[:12]
        return f"PATIENT_{digest.upper()}"

    def scrub_text(self, text: str) -> str:
        """Sanitizes text by stripping out PII and sensitive identifiers.
        
        Args:
            text: Raw input narrative.
            
        Returns:
            Sanitized text with PII replaced by bracketed tokens.
        """
        if not text or not isinstance(text, str):
            return ""
        sanitized = text
        for pattern, replacement in self.patterns:
            sanitized = pattern.sub(replacement, sanitized)
        # Normalize excessive whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        author_col: str = "author",
        text_cols: Optional[List[str]] = None,
        drop_raw_author: bool = True
    ) -> pd.DataFrame:
        """Applies pseudonymization and scrubbing to a full pandas DataFrame.
        
        Args:
            df: DataFrame containing social health posts.
            author_col: Name of column with raw author names.
            text_cols: List of column names containing text to scrub (defaults to ['title', 'body', 'text']).
            drop_raw_author: Whether to drop the original raw author column.
            
        Returns:
            Anonymized DataFrame with 'patient_id' column added.
        """
        anonymized_df = df.copy()
        if text_cols is None:
            text_cols = [col for col in ["title", "body", "text", "content"] if col in anonymized_df.columns]

        if author_col in anonymized_df.columns:
            anonymized_df["patient_id"] = anonymized_df[author_col].apply(self.hash_author_id)
            if drop_raw_author:
                anonymized_df.drop(columns=[author_col], inplace=True)
        elif "patient_id" not in anonymized_df.columns:
            anonymized_df["patient_id"] = [f"PATIENT_{i:04d}" for i in range(len(anonymized_df))]

        for col in text_cols:
            if col in anonymized_df.columns:
                anonymized_df[col] = anonymized_df[col].astype(str).apply(self.scrub_text)

        return anonymized_df
