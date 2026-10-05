# Clinical Case Study: PATIENT_015F6DE9AE3F

**Predicted Trajectory:** `Long Delay (> 1 Year)` (Risk Score: **100.0%**)
**Ground Truth Label:** `Long Delay (> 1 Year)` (Concordant: **True**)
**Pre-Diagnostic Odyssey Span:** 1524.7 days across 5 longitudinal posts.

---
## Longitudinal Patient Narrative & Attention Weights

| Post # | Day | Gap (Days) | Attention Weight (α) | Narrative Summary & Key Linguistic Signal |
|:------:|:---:|:----------:|:--------------------:|:------------------------------------------|
| 1 | 0 | 0 | `0.336` ⭐ **[PEAK]** | Unexplained feeling like I have the flu for months and burning neuropathy for months, doctors clueless I've been dealing with feeling like I... |
| 2 | 396 | 396 | `0.304` | Doctor says it's in my head / just anxiety Saw another doctor today. Doctor dismissed my symptoms and said my blood work was completely norm... |
| 3 | 805 | 408 | `0.251` | Years of progressive symptoms (mouth sores and morning stiffness lasting hours) - still no answers It has been so long since this started. N... |
| 4 | 1062 | 257 | `0.103` | Years of progressive symptoms (extremely dry eyes like sandpaper and muscle weakness) - still no answers It has been so long since this star... |
| 5 | 1525 | 463 | `0.006` | Years of progressive symptoms (Raynaud's fingers turning blue and white and morning stiffness lasting hours) - still no answers It has been ... |

---
## Explainability & Feature Attribution (SHAP)

| Predictive Feature | Observed Value | SHAP Impact | Clinical Interpretation |
|:-------------------|:--------------:|:-----------:|:------------------------|
| `diagnostic_tests_count` | `0.0` | `+0.065` | 🔺 Increases Delay Risk |
| `max_frustration` | `5.714` | `+0.058` | 🔺 Increases Delay Risk |
| `posts_with_dismissal_ratio` | `0.6` | `+0.056` | 🔺 Increases Delay Risk |
| `mean_net_polarity` | `-2.529` | `+0.051` | 🔺 Increases Delay Risk |
| `dismissal_rate` | `2.027` | `+0.048` | 🔺 Increases Delay Risk |
| `mean_frustration` | `2.529` | `+0.046` | 🔺 Increases Delay Risk |

---
## Qualitative Discussion & Takeaway
> **Clinical Takeaway**: This patient exhibits hallmark diagnostic odyssey markers, characterized by multiple medical dismissal events ('all in your head / just anxiety'), prolonged inter-post latency, and persistent multisystem complaints. The longitudinal sequence model placed peak attention on the initial dismissal encounter, demonstrating early detectability of delay trajectories.