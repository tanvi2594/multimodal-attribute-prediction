"""Per-attribute LightGBM multiclass heads over fused embeddings."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


class AttributeHead:
    """One LightGBM multiclass classifier for a single product attribute."""

    def __init__(self, attribute: str, params: dict):
        self.attribute = attribute
        self.params = dict(params)
        self.early_stopping = self.params.pop("early_stopping_rounds", 100)
        self.label_encoder = LabelEncoder()
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, x_train, y_train, x_val, y_val) -> dict:
        y_train_enc = self.label_encoder.fit_transform(y_train)
        # Map unseen val labels to a sentinel so metrics are still computable.
        known = set(self.label_encoder.classes_)
        val_mask = np.array([y in known for y in y_val])
        y_val_enc = self.label_encoder.transform(np.asarray(y_val)[val_mask])
        x_val_known = x_val[val_mask]

        self.model = lgb.LGBMClassifier(
            num_class=len(self.label_encoder.classes_), **self.params
        )
        self.model.fit(
            x_train, y_train_enc,
            eval_set=[(x_val_known, y_val_enc)],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
        )

        preds = self.model.predict(x_val_known)
        return {
            "attribute": self.attribute,
            "n_classes": int(len(self.label_encoder.classes_)),
            "val_accuracy": float(accuracy_score(y_val_enc, preds)),
            "val_macro_f1": float(f1_score(y_val_enc, preds, average="macro")),
            "best_iteration": int(self.model.best_iteration_ or self.params.get("n_estimators", 0)),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(self.model.predict(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def save(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "label_encoder": self.label_encoder, "attribute": self.attribute},
            out_dir / f"{self.attribute}.joblib",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AttributeHead":
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.attribute = blob["attribute"]
        obj.model = blob["model"]
        obj.label_encoder = blob["label_encoder"]
        return obj
