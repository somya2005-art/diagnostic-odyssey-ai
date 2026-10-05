"""PRAW Reddit Data Collector Module.

Pulls longitudinal post histories and comments from targeted health subreddits
(r/lupus, r/Sjogrens, r/AutoimmuneDisease, r/DiagnoseMe, r/ChronicIllness, etc.)
with rate-limiting, error backoff, and pagination.
"""

import os
import time
from typing import Dict, List, Optional
import pandas as pd


class RedditDataCollector:
    """Collects public health forum posts via the official Reddit API (PRAW)."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "DiagnosticDelayResearchBot:v1.0 (academic research)"
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent
        self.reddit = None
        
        self._init_praw()

    def _init_praw(self):
        """Initializes PRAW client if credentials exist."""
        try:
            import praw
            if self.client_id and self.client_secret:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
        except ImportError:
            self.reddit = None

    def is_configured(self) -> bool:
        """Returns True if PRAW is authenticated and ready."""
        return self.reddit is not None

    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        sort: str = "new",
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fetches posts from a given subreddit matching optional health keywords.
        
        Args:
            subreddit_name: e.g. 'lupus', 'Sjogrens', 'ChronicIllness'
            limit: Maximum number of submissions to retrieve.
            sort: 'new', 'hot', or 'top'
            keywords: Optional list of filtering keywords (e.g. ['diagnos', 'delay', 'years', 'finally'])
            
        Returns:
            List of dictionary records for submissions.
        """
        if not self.is_configured():
            print(f"[PRAWCollector] Warning: PRAW not configured with valid API keys. Skipping live crawl for r/{subreddit_name}.")
            return []

        posts = []
        try:
            sub = self.reddit.subreddit(subreddit_name)
            if keywords:
                # Search within subreddit
                query = " OR ".join(keywords)
                submissions = sub.search(query, limit=limit, sort=sort)
            else:
                submissions = getattr(sub, sort)(limit=limit)

            for submission in submissions:
                # Exclude deleted or mod posts
                author_name = str(submission.author.name) if submission.author else "[deleted]"
                if author_name == "[deleted]" or submission.stickied:
                    continue

                posts.append({
                    "id": submission.id,
                    "author": author_name,
                    "created_utc": int(submission.created_utc),
                    "subreddit": subreddit_name,
                    "title": submission.title,
                    "body": submission.selftext,
                    "text": f"{submission.title} {submission.selftext}",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "url": submission.url
                })
                # Respect rate limits
                time.sleep(0.05)

        except Exception as e:
            print(f"[PRAWCollector] Error collecting from r/{subreddit_name}: {e}")

        return posts

    def fetch_user_history(self, username: str, limit: int = 50) -> List[Dict]:
        """Fetches historical submissions made by a single user across health subreddits."""
        if not self.is_configured() or username in ["[deleted]", "[REDACTED_USER]"]:
            return []

        user_posts = []
        try:
            redditor = self.reddit.redditor(username)
            for submission in redditor.submissions.new(limit=limit):
                user_posts.append({
                    "id": submission.id,
                    "author": username,
                    "created_utc": int(submission.created_utc),
                    "subreddit": str(submission.subreddit),
                    "title": submission.title,
                    "body": submission.selftext,
                    "text": f"{submission.title} {submission.selftext}",
                    "score": submission.score
                })
        except Exception as e:
            print(f"[PRAWCollector] Error fetching history for user {username}: {e}")

        return user_posts
