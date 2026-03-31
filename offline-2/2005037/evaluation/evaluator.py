import numpy as np
from metrics.classification_metrics import (
    compute_accuracy,
    compute_f1,
    compute_auroc
)


class Evaluator:
    """
    Unified evaluation pipeline for classification models.
    """

    def __init__(self, metrics):
        """
        Parameters
        ----------
        metrics : list of str
            Supported metrics: 'accuracy', 'f1', 'auroc'
        """
        self.metrics = metrics

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate a trained model.

        Returns
        -------
        results : dict
            Metric name -> score
        """
        results = {}

        y_pred = model.predict(X_test)

        if "accuracy" in self.metrics:
            results["accuracy"] = compute_accuracy(y_test, y_pred)

        if "f1" in self.metrics:
            results["f1"] = compute_f1(y_test, y_pred)

        if "auroc" in self.metrics:
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "Model must implement predict_proba() for AUROC"
                )

            y_proba = model.predict_proba(X_test)
            n_classes = len(np.unique(y_test))

            results["auroc"] = compute_auroc(
                y_test, y_proba, n_classes
            )

        return results
