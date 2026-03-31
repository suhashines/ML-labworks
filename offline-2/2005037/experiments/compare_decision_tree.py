from data.loader import DatasetLoader
from models.decision_tree import DecisionTree
from evaluation.evaluator import Evaluator
from evaluation.sklearn_baselines import SklearnDecisionTree
from config import (
    DECISION_TREE_CONFIG,
    DATASET_CONFIG,
    EVALUATION_CONFIG,
    RANDOM_SEED
)


def run_experiment(dataset_name):
    print(f"\n=== Dataset: {dataset_name.upper()} ===")

    # Load data
    loader = DatasetLoader(
        test_size=DATASET_CONFIG[dataset_name]["test_size"],
        random_state=RANDOM_SEED
    )
    X_train, X_test, y_train, y_test = loader.load_and_split(dataset_name)

    # Initialize evaluator
    evaluator = Evaluator(EVALUATION_CONFIG["metrics"])

    # -----------------------
    # Custom Decision Tree
    # -----------------------
    custom_dt = DecisionTree(
        max_depth=DECISION_TREE_CONFIG["max_depth"],
        min_samples_split=DECISION_TREE_CONFIG["min_samples_split"],
        criterion=DECISION_TREE_CONFIG["criterion"],
        random_state=RANDOM_SEED
    )

    custom_dt.fit(X_train, y_train)
    custom_results = evaluator.evaluate(
        custom_dt, X_test, y_test
    )

    # -----------------------
    # Sklearn Decision Tree
    # -----------------------
    sklearn_dt = SklearnDecisionTree(
        max_depth=DECISION_TREE_CONFIG["max_depth"],
        min_samples_split=DECISION_TREE_CONFIG["min_samples_split"],
        criterion=DECISION_TREE_CONFIG["criterion"],
        random_state=RANDOM_SEED
    )

    sklearn_dt.fit(X_train, y_train)
    sklearn_results = evaluator.evaluate(
        sklearn_dt, X_test, y_test
    )

    # -----------------------
    # Results
    # -----------------------
    print("\nCustom Decision Tree:")
    for k, v in custom_results.items():
        print(f"{k:10s}: {v:.4f}")

    print("\nSklearn Decision Tree:")
    for k, v in sklearn_results.items():
        print(f"{k:10s}: {v:.4f}")


if __name__ == "__main__":
    run_experiment("iris")
    run_experiment("wine")
