"""
Central configuration file for Assignment 2:
Decision Trees, Random Forests, and Extra Trees
"""

# =========================
# Global Settings
# =========================

RANDOM_SEED = 42


# =========================
# Dataset Settings
# =========================

DATASET_CONFIG = {
    "iris": {
        "test_size": 0.2
    },
    "wine": {
        "test_size": 0.2
    }
}


# =========================
# Decision Tree Hyperparameters
# =========================

DECISION_TREE_CONFIG = {
    "max_depth": None,            # None means grow until pure or min_samples_split
    "min_samples_split": 2,
    "criterion": "entropy"            # or "entropy"
}


# =========================
# Random Forest Hyperparameters
# =========================

RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": "sqrt",        # sqrt, log2, or int
    "bootstrap": True
}


# =========================
# Extra Trees Hyperparameters
# =========================

EXTRA_TREES_CONFIG = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "bootstrap": False             # Extra Trees typically do not bootstrap
}


# =========================
# Evaluation Settings
# =========================

EVALUATION_CONFIG = {
    "metrics": ["accuracy", "f1", "auroc"]
}
