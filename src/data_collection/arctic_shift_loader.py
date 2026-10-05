"""Arctic Shift & Historical Dump Ingestion Module.

Loads and streams historical JSONL/JSON dumps from Arctic Shift or Academic Torrents,
filtering by health subreddits, date ranges, and relevant keyword queries.
"""

import json
import gzip
import os
from typing import Dict, Iterator, List, Optional
import pandas as pd


class ArcticShiftLoader:
    """Parses and filters bulk historical Reddit submissions from Arctic Shift / Pushshift dumps."""

    def __init__(self, target_subreddits: Optional[List[str]] = None):
        self.target_subreddits = [s.lower().replace("r/", "") for s in target_subreddits] if target_subreddits else []

    def stream_dump_file(
        self,
        file_path: str,
        max_records: Optional[int] = None
    ) -> Iterator[Dict]:
        """Streams records one by one from a .jsonl or .jsonl.gz file.
        
        Args:
            file_path: Path to the dump file.
            max_records: Maximum number of records to yield.
            
        Yields:
            Parsed record dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dump file not found: {file_path}")

        open_fn = gzip.open if file_path.endswith(".gz") else open
        mode = "rt" if file_path.endswith(".gz") else "r"

        count = 0
        with open_fn(file_path, mode, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    subreddit = record.get("subreddit", "").lower()
                    if self.target_subreddits and subreddit not in self.target_subreddits:
                        continue

                    # Filter out removed / deleted
                    body = record.get("selftext", "") or record.get("body", "")
                    if body in ["[removed]", "[deleted]"] or record.get("author") in ["[deleted]"]:
                        continue

                    yield {
                        "id": record.get("id"),
                        "author": record.get("author"),
                        "created_utc": int(record.get("created_utc", 0)),
                        "subreddit": record.get("subreddit"),
                        "title": record.get("title", ""),
                        "body": body,
                        "text": f"{record.get('title', '')} {body}".strip(),
                        "score": record.get("score", 0)
                    }
                    count += 1
                    if max_records and count >= max_records:
                        break
                except json.JSONDecodeError:
                    continue

    def load_dump_to_dataframe(
        self,
        file_path: str,
        max_records: Optional[int] = 5000
    ) -> pd.DataFrame:
        """Loads streamed records directly into a Pandas DataFrame."""
        records = list(self.stream_dump_file(file_path, max_records=max_records))
        return pd.DataFrame(records)
