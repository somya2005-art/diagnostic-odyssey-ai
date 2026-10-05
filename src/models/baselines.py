"""Baseline Models for Diagnostic Delay Prediction.

Provides standard classical ML baselines:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Tabular Structured Features + Random Forest
4. Combined Multi-Modal Gradient Boosting Classifier
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold


class BaselineModelSuite:
    """Manages training and evaluation of classical ML baselines."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Logistic_Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state
            ),
            "Linear_SVM": LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                dual="auto"
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                class_weight="balanced",
                random_state=random_state
            ),
            "Gradient_Boosting": HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                class_weight="balanced",
                random_state=random_state
            )
        }
        self.fitted_models: Dict[str, Any] = {}

    def fit_and_evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> pd.DataFrame:
        """Evaluates all baseline models using Stratified K-Fold cross-validation.
        
        Args:
            X: Feature matrix of shape (N, D).
            y: Binary target array of shape (N,).
            n_splits: Number of cross-validation folds.
            
        Returns:
            Summary DataFrame comparing metrics across baselines.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results = []

        for model_name, model in self.models.items():
            accs, precs, recs, f1s, rocs, prs = [], [], [], [], [], []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_val)

                # Probabilities / Decision scores for ROC-AUC
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, "decision_function"):
                    df_scores = model.decision_function(X_val)
                    # Min-max scale decision function to [0, 1] for metric computation
                    y_prob = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-6)
                else:
                    y_prob = y_pred

                accs.append(accuracy_score(y_val, y_pred))
                precs.append(precision_score(y_val, y_pred, zero_division=0))
                recs.append(recall_score(y_val, y_pred, zero_division=0))
                f1s.append(f1_score(y_val, y_pred, zero_division=0))
                rocs.append(roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5)
                prs.append(average_precision_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5)

            results.append({
                "Model": model_name,
                "Accuracy": round(float(np.mean(accs)), 4),
                "Precision": round(float(np.mean(precs)), 4),
                "Recall": round(float(np.mean(recs)), 4),
                "F1_Score": round(float(np.mean(f1s)), 4),
                "ROC_AUC": round(float(np.mean(rocs)), 4),
                "PR_AUC": round(float(np.mean(prs)), 4)
            })

        # Fit all models on full dataset for downstream inference / SHAP
        for model_name, model in self.models.items():
            model.fit(X, y)
            self.fitted_models[model_name] = model

        return pd.DataFrame(results)
