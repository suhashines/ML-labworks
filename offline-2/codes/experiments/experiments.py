import numpy as np
import pandas as pd

from data.loader import DatasetLoader
from evaluation.evaluator import Evaluator
from evaluation.sklearn_baselines import (
    SklearnDecisionTree,
    SklearnRandomForest,
    SklearnExtraTrees
)

from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.extra_trees import ExtraTrees

from config import (
    DATASET_CONFIG,
    DECISION_TREE_CONFIG,
    RANDOM_FOREST_CONFIG,
    EXTRA_TREES_CONFIG,
    EVALUATION_CONFIG,
    RANDOM_SEED
)


# ==================================================
# Model Factory
# ==================================================

def build_model(model_name, impl="custom", **params):

    if impl == "custom":

        if model_name == "dt":
            return DecisionTree(**params)

        if model_name == "rf":
            return RandomForest(**params)

        if model_name == "et":
            return ExtraTrees(**params)

    elif impl == "sklearn":

        if model_name == "dt":
            return SklearnDecisionTree(**params)

        if model_name == "rf":
            return SklearnRandomForest(**params)

        if model_name == "et":
            return SklearnExtraTrees(**params)

    raise ValueError("Invalid model configuration")


# ==================================================
# Core Experiment Runner
# ==================================================

def run_single_experiment(
    model_name,
    dataset,
    impl="custom",
    model_params=None
):
    """
    Train + evaluate a single model.
    """

    if model_params is None:
        model_params = {}

    # Load data
    loader = DatasetLoader(
        test_size=DATASET_CONFIG[dataset]["test_size"],
        random_state=RANDOM_SEED
    )

    X_train, X_test, y_train, y_test = loader.load_and_split(dataset)

    # Build model
    model = build_model(
        model_name,
        impl,
        **model_params
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    evaluator = Evaluator(EVALUATION_CONFIG["metrics"])
    results = evaluator.evaluate(model, X_test, y_test)

    return results


# ==================================================
# Compare Custom vs Sklearn
# ==================================================

def compare_models(model_name, dataset):
    """
    Compare custom vs sklearn implementation.
    """

    if model_name == "dt":
        params = DECISION_TREE_CONFIG

    elif model_name == "rf":
        params = RANDOM_FOREST_CONFIG

    elif model_name == "et":
        params = EXTRA_TREES_CONFIG

    else:
        raise ValueError("Unknown model")

    custom_results = run_single_experiment(
        model_name,
        dataset,
        impl="custom",
        model_params=params
    )

    sklearn_results = run_single_experiment(
        model_name,
        dataset,
        impl="sklearn",
        model_params=params
    )

    return {
        "custom": custom_results,
        "sklearn": sklearn_results
    }


# ==================================================
# Compare All Models
# ==================================================

def compare_all_models(dataset):
    """
    Compare DT, RF, ET (custom vs sklearn)
    """

    results = {}

    for model in ["dt", "rf", "et"]:
        results[model] = compare_models(model, dataset)

    return results


# ==================================================
# Hyperparameter Sweep
# ==================================================

def hyperparameter_sweep(
    model_name,
    dataset,
    param_name,
    param_values,
    impl="custom"
):
    """
    Run experiments for different hyperparameter values.
    """

    records = []

    # Select base config
    if model_name == "dt":
        base_params = DECISION_TREE_CONFIG.copy()

    elif model_name == "rf":
        base_params = RANDOM_FOREST_CONFIG.copy()

    elif model_name == "et":
        base_params = EXTRA_TREES_CONFIG.copy()

    else:
        raise ValueError("Invalid model")

    for value in param_values:

        params = base_params.copy()
        params[param_name] = value

        result = run_single_experiment(
            model_name,
            dataset,
            impl=impl,
            model_params=params
        )

        record = {
            "model": model_name,
            "impl": impl,
            "dataset": dataset,
            "param": param_name,
            "value": value
        }

        record.update(result)
        records.append(record)

    return pd.DataFrame(records)


# ==================================================
# Bias-Variance Study
# ==================================================

def bias_variance_study(
    model_name,
    dataset,
    param_name,
    param_values,
    n_runs=5
):
    """
    Repeat experiments to analyze variance.
    """

    records = []

    for run in range(n_runs):

        seed = RANDOM_SEED + run
        np.random.seed(seed)

        df = hyperparameter_sweep(
            model_name,
            dataset,
            param_name,
            param_values,
            impl="custom"
        )

        df["run"] = run
        records.append(df)

    return pd.concat(records, ignore_index=True)


# ==================================================
# Learning Curve
# ==================================================

def learning_curve(
    model_name,
    dataset,
    train_sizes,
    impl="custom"
):
    """
    Analyze performance vs training set size.
    """

    loader = DatasetLoader(
        test_size=DATASET_CONFIG[dataset]["test_size"],
        random_state=RANDOM_SEED
    )

    X, y = loader.load(dataset)

    records = []

    for size in train_sizes:

        n = int(len(X) * size)

        idxs = np.random.permutation(len(X))[:n]

        X_sub = X[idxs]
        y_sub = y[idxs]

        X_train, X_test, y_train, y_test = loader.train_test_split(
            X_sub, y_sub
        )

        model = build_model(
            model_name,
            impl,
            **get_base_params(model_name)
        )

        model.fit(X_train, y_train)

        evaluator = Evaluator(EVALUATION_CONFIG["metrics"])
        result = evaluator.evaluate(model, X_test, y_test)

        record = {
            "train_fraction": size,
            "model": model_name,
            "impl": impl,
            "dataset": dataset
        }

        record.update(result)
        records.append(record)

    return pd.DataFrame(records)


# ==================================================
# Utilities
# ==================================================

def get_base_params(model_name):

    if model_name == "dt":
        return DECISION_TREE_CONFIG.copy()

    if model_name == "rf":
        return RANDOM_FOREST_CONFIG.copy()

    if model_name == "et":
        return EXTRA_TREES_CONFIG.copy()

    raise ValueError("Invalid model")
