# Clinical Case Study: PATIENT_042C7F60E4E9

**Predicted Trajectory:** `Short Delay (<= 1 Year)` (Risk Score: **0.0%**)
**Ground Truth Label:** `Short Delay (<= 1 Year)` (Concordant: **True**)
**Pre-Diagnostic Odyssey Span:** 156.6 days across 3 longitudinal posts.

---
## Longitudinal Patient Narrative & Attention Weights

| Post # | Day | Gap (Days) | Attention Weight (α) | Narrative Summary & Key Linguistic Signal |
|:------:|:---:|:----------:|:--------------------:|:------------------------------------------|
| 1 | 0 | 0 | `0.349` | Sudden onset of butterfly rash on cheeks after sun exposure and feeling like I have the flu for months - what should I ask my doctor? Hi eve... |
| 2 | 109 | 109 | `0.402` ⭐ **[PEAK]** | Update: Doctor ordered ANA and rheumatology referral Had my doctor appointment today. PCP listened carefully to my joint swelling and suspec... |
| 3 | 157 | 48 | `0.249` | Update: Doctor ordered ANA and rheumatology referral Had my doctor appointment today. My doctor immediately ordered an ANA panel and referre... |

---
## Explainability & Feature Attribution (SHAP)

| Predictive Feature | Observed Value | SHAP Impact | Clinical Interpretation |
|:-------------------|:--------------:|:-----------:|:------------------------|
| `diagnostic_tests_count` | `3.0` | `+0.065` | 🔺 Increases Delay Risk |
| `max_frustration` | `0.0` | `+0.058` | 🔺 Increases Delay Risk |
| `posts_with_dismissal_ratio` | `0.0` | `+0.056` | 🔺 Increases Delay Risk |
| `mean_net_polarity` | `0.952` | `+0.051` | 🔺 Increases Delay Risk |
| `dismissal_rate` | `0.0` | `+0.048` | 🔺 Increases Delay Risk |
| `mean_frustration` | `0.0` | `+0.046` | 🔺 Increases Delay Risk |

---
## Qualitative Discussion & Takeaway
> **Clinical Takeaway**: This patient shows a rapid diagnostic pathway with prompt clinical validation and immediate antibody testing. The absence of gaslighting markers and short inter-post intervals reliably distinguished this trajectory from prolonged delay odysseys.