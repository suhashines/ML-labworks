import numpy as np
from collections import Counter
from abc import ABC, abstractmethod


class TreeNode:
    """
    A single node in the decision tree.
    """

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        class_counts=None,
        class_probs=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_counts = class_counts
        self.class_probs = class_probs

    def is_leaf(self):
        return self.value is not None


class BaseTree(ABC):
    """
    Abstract base class for all tree-based classifiers.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.root = None

        if random_state is not None:
            np.random.seed(random_state)

    # =========================
    # Public API
    # =========================

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array(
            [self._predict_sample(x, self.root) for x in X]
        )

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], self.n_classes_))

        for i, x in enumerate(X):
            leaf = self._traverse_tree(x, self.root)
            for cls, p in leaf.class_probs.items():
                proba[i, cls] = p

        return proba

    # =========================
    # Tree Construction
    # =========================

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (
            num_labels == 1
            or num_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return self._create_leaf(y)

        feature_indices = self._get_feature_indices(num_features)
        best_feature, best_threshold = self._best_split(
            X, y, feature_indices
        )

        if best_feature is None:
            return self._create_leaf(y)

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold

        left_child = self._grow_tree(
            X[left_idxs], y[left_idxs], depth + 1
        )
        right_child = self._grow_tree(
            X[right_idxs], y[right_idxs], depth + 1
        )

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def _create_leaf(self, y):
        counter = Counter(y)
        total = sum(counter.values())

        class_probs = {
            cls: count / total for cls, count in counter.items()
        }

        most_common_class = counter.most_common(1)[0][0]

        return TreeNode(
            value=most_common_class,
            class_counts=dict(counter),
            class_probs=class_probs
        )

    # =========================
    # Prediction Helpers
    # =========================

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    # =========================
    # Split Logic
    # =========================

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature in feature_indices:
            values = np.unique(X[:, feature])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                gain = self._information_gain(
                    y, X[:, feature], threshold
                )

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, feature_column, threshold):
        parent_impurity = self._impurity(y)

        left = feature_column <= threshold
        right = feature_column > threshold

        if np.sum(left) == 0 or np.sum(right) == 0:
            return 0

        n = len(y)
        n_l, n_r = np.sum(left), np.sum(right)

        child_impurity = (
            (n_l / n) * self._impurity(y[left])
            + (n_r / n) * self._impurity(y[right])
        )

        return parent_impurity - child_impurity

    # =========================
    # Impurity
    # =========================

    def _impurity(self, y):
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError("Unknown criterion")

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    # =========================
    # Hooks
    # =========================

    @abstractmethod
    def _get_feature_indices(self, num_features):
        pass
