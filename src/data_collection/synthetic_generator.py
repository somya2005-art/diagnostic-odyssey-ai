"""Synthetic Patient Timeline Generator for Diagnostic Delay NLP Research.

Generates realistic longitudinal social health post trajectories simulating
the 'Diagnostic Odyssey', medical dismissal events, symptom clusters,
inter-post intervals, and self-reported diagnostic outcomes.
"""

import random
import time
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class SyntheticPatientGenerator:
    """Generates realistic synthetic patient post sequences for benchmark validation."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.subreddits = [
            "r/lupus", "r/Sjogrens", "r/AutoimmuneDisease", 
            "r/DiagnoseMe", "r/ChronicIllness", "r/spoonies", "r/UndiagnosedIllness"
        ]
        
        # Clinical symptom vocabularies by cluster
        self.symptoms = {
            "musculoskeletal": ["severe joint pain in hands and knees", "morning stiffness lasting hours", "swollen knuckles", "unbearable wrist aching", "muscle weakness"],
            "systemic": ["debilitating fatigue", "unexplained low-grade fevers", "feeling like I have the flu for months", "severe brain fog", "night sweats"],
            "mucocutaneous": ["butterfly rash on cheeks after sun exposure", "malar rash across nose", "extremely dry eyes like sandpaper", "dry mouth and difficulty swallowing", "mouth sores", "Raynaud's fingers turning blue and white"],
            "neurological": ["tingling in feet and fingers", "burning neuropathy", "dizziness and POTS-like symptoms", "memory lapses", "migraines"]
        }
        
        # Medical dismissal patterns (strong signal for long diagnostic delay)
        self.dismissal_phrases = [
            "My primary care doctor told me it's just anxiety and to do yoga.",
            "Doctor dismissed my symptoms and said my blood work was completely normal.",
            "Was told it's all in my head and referred to psychiatry instead of rheumatology.",
            "Doctor said my joint pain is just stress from work and refused to run an autoimmune panel.",
            "The rheumatologist barely looked at my butterfly rash and told me I was overthinking.",
            "Doctor said you are too young and healthy to have an autoimmune disease.",
            "Told to stop googling symptoms because I'm just making myself anxious.",
            "Third doctor who told me labs are fine so there is nothing wrong with me."
        ]
        
        # Medical validation / responsive encounters (more common in short delay)
        self.validation_phrases = [
            "My doctor immediately ordered an ANA panel and referred me to rheumatology.",
            "PCP listened carefully to my joint swelling and suspected an autoimmune condition.",
            "Got referred to a specialist right away after abnormal bloodwork.",
            "Doctor took my symptoms seriously and started a workup."
        ]
        
        # Emotional tones
        self.frustrated_tones = [
            "I feel so hopeless and ignored by the healthcare system.",
            "Crying in my car after another useless appointment. Does anyone else feel gaslit?",
            "Losing my mind trying to get someone to believe how sick I feel.",
            "Exhausted from fighting doctors to be taken seriously."
        ]
        
        self.inquisitive_tones = [
            "Has anyone experienced something similar? Any advice on what tests to ask for?",
            "Looking for recommendations for a good rheumatologist who actually listens.",
            "Could this be lupus or Sjogrens? Here are my symptoms..."
        ]

    def _generate_short_delay_patient(self, patient_idx: int) -> List[Dict]:
        """Generates a patient timeline resulting in short diagnostic delay (<= 1.0 yr)."""
        author = f"user_short_{patient_idx:04d}"
        num_posts = random.randint(3, 5)
        
        # Total span between 1 to 10 months (30 to 300 days)
        span_days = random.randint(45, 270)
        start_time = 1609459200 + random.randint(0, 31536000)  # Around 2021-2022
        
        # Generate timestamps spread over the span
        intervals = sorted([random.uniform(0, span_days) for _ in range(num_posts - 1)]) + [span_days]
        timestamps = [start_time + int(day * 86400) for day in intervals]
        
        posts = []
        condition = random.choice(["Lupus", "Sjögren's syndrome", "Autoimmune arthritis"])
        
        for i in range(num_posts):
            sub = random.choice(self.subreddits)
            ts = timestamps[i]
            
            if i == 0:
                # Early symptom presentation
                s1 = random.choice(self.symptoms["mucocutaneous"])
                s2 = random.choice(self.symptoms["systemic"])
                title = f"Sudden onset of {s1} and {s2} - what should I ask my doctor?"
                body = f"Hi everyone, over the past few weeks I have noticed {s1}. Also having {s2}. {random.choice(self.inquisitive_tones)}"
            elif i < num_posts - 1:
                # Progression / doctor visit
                val_phrase = random.choice(self.validation_phrases)
                s3 = random.choice(self.symptoms["musculoskeletal"])
                title = "Update: Doctor ordered ANA and rheumatology referral"
                body = f"Had my doctor appointment today. {val_phrase} Also noticed some {s3}. Hoping for clear results soon."
            else:
                # Final post with diagnosis milestone (Short duration)
                duration_months = max(1, int(span_days / 30))
                title = f"Finally diagnosed with {condition}!"
                body = f"After a stressful couple of months, I was officially diagnosed after {duration_months} months. Starting hydroxychloroquine next week. Thank you all for the support!"
            
            posts.append({
                "author": author,
                "created_utc": ts,
                "subreddit": sub,
                "title": title,
                "body": body,
                "text": f"{title} {body}",
                "true_delay_category": 0,  # Short delay
                "true_duration_years": round(span_days / 365.0, 2)
            })
            
        return posts

    def _generate_long_delay_patient(self, patient_idx: int) -> List[Dict]:
        """Generates a patient timeline resulting in long diagnostic delay (> 1.0 yr, e.g. 2-6 years)."""
        author = f"user_long_{patient_idx:04d}"
        num_posts = random.randint(4, 8)
        
        # Total span between 1.5 to 6 years (550 to 2200 days)
        delay_years = random.uniform(1.8, 5.5)
        span_days = int(delay_years * 365)
        start_time = 1546300800 + random.randint(0, 31536000)  # Around 2019-2020
        
        intervals = sorted([random.uniform(0, span_days) for _ in range(num_posts - 1)]) + [span_days]
        timestamps = [start_time + int(day * 86400) for day in intervals]
        
        posts = []
        condition = random.choice(["Systemic Lupus Erythematosus", "Sjögren's Syndrome", "Undifferentiated Connective Tissue Disease (UCTD)"])
        
        for i in range(num_posts):
            sub = random.choice(self.subreddits)
            ts = timestamps[i]
            
            if i == 0:
                s1 = random.choice(self.symptoms["systemic"])
                s2 = random.choice(self.symptoms["neurological"])
                title = f"Unexplained {s1} and {s2} for months, doctors clueless"
                body = f"I've been dealing with {s1} and constant {s2}. Every blood test comes back normal. {random.choice(self.inquisitive_tones)}"
            elif i == 1:
                # First dismissal
                d_phrase = random.choice(self.dismissal_phrases)
                title = "Doctor says it's in my head / just anxiety"
                body = f"Saw another doctor today. {d_phrase} {random.choice(self.frustrated_tones)}"
            elif i < num_posts - 1:
                # Multi-system flare, secondary dismissal, high frustration
                s_multi = random.choice(self.symptoms["mucocutaneous"]) + " and " + random.choice(self.symptoms["musculoskeletal"])
                d_phrase2 = random.choice(self.dismissal_phrases)
                f_tone = random.choice(self.frustrated_tones)
                title = f"Years of progressive symptoms ({s_multi}) - still no answers"
                body = f"It has been so long since this started. Now having {s_multi}. {d_phrase2} {f_tone} How do you keep advocating for yourself when doctors dismiss everything?"
            else:
                # Final post with diagnosis milestone (Long duration)
                years_int = int(round(delay_years))
                title = f"Finally diagnosed with {condition} after {years_int} years of medical gaslighting"
                body = f"I wanted to post an update because it took me {years_int} years to get diagnosed. Saw 5 different specialists who dismissed me before a rheumatologist finally ran the right antibodies. Don't give up!"
            
            posts.append({
                "author": author,
                "created_utc": ts,
                "subreddit": sub,
                "title": title,
                "body": body,
                "text": f"{title} {body}",
                "true_delay_category": 1,  # Long delay
                "true_duration_years": round(delay_years, 2)
            })
            
        return posts

    def generate_dataset(
        self,
        num_patients: int = 150,
        long_delay_ratio: float = 0.55
    ) -> pd.DataFrame:
        """Generates a complete multi-patient longitudinal dataset.
        
        Args:
            num_patients: Total number of unique patients to generate.
            long_delay_ratio: Proportion of patients with long diagnostic delay.
            
        Returns:
            DataFrame of all posts from all patients.
        """
        all_posts = []
        num_long = int(num_patients * long_delay_ratio)
        num_short = num_patients - num_long
        
        for idx in range(num_short):
            posts = self._generate_short_delay_patient(idx)
            all_posts.extend(posts)
            
        for idx in range(num_long):
            posts = self._generate_long_delay_patient(idx + num_short)
            all_posts.extend(posts)
            
        df = pd.DataFrame(all_posts)
        # Randomize order across patients to simulate realistic scraping stream
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df
