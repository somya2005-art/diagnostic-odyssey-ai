# Diagnostic Odyssey AI

Predicting Diagnostic Delay Signals from Patient-Authored Text via Longitudinal NLP.

## Live Demo
Deployed at: https://diagnostic-odyssey-ai.onrender.com *(link active after deployment)*

## Local Setup

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8080
```

## Deploy to Render (free)
1. Push this repo to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click Deploy

## Tech Stack
- PyTorch · Sentence-BERT · BiLSTM + Bahdanau Attention · SHAP · LIME
- AoIR Ethics Compliant · Longitudinal NLP · Explainable AI
