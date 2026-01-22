import numpy as np
from models.base_tree import BaseTree


class ExtraTree(BaseTree):
    """
    Single Extremely Randomized Tree.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        random_state=None
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        self.max_features = max_features

    def _get_feature_indices(self, num_features):

        max_feat = self._resolve_max_features(num_features)

        return np.random.choice(
            num_features,
            max_feat,
            replace=False
        )

    def _resolve_max_features(self, n_features):

        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))

        if self.max_features == "log2":
            return int(np.log2(n_features))

        if isinstance(self.max_features, int):
            return self.max_features

        return n_features

    def _best_split(self, X, y, feature_indices):

        best_feature = None
        best_threshold = None

        for feature in feature_indices:

            col = X[:, feature]

            min_val = np.min(col)
            max_val = np.max(col)

            if min_val == max_val:
                continue

            threshold = np.random.uniform(
                min_val, max_val
            )

            best_feature = feature
            best_threshold = threshold

            break

        return best_feature, best_threshold


class ExtraTrees:
    """
    Extremely Randomized Trees ensemble.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=False,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):

        self.trees = []
        self.n_classes_ = len(np.unique(y))

        for i in range(self.n_estimators):

            X_sample, y_sample = self._sample(X, y)

            tree = ExtraTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self._get_seed(i)
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):

        all_probs = np.zeros(
            (X.shape[0], self.n_classes_)
        )

        for tree in self.trees:
            all_probs += tree.predict_proba(X)

        return all_probs / self.n_estimators

    # =========================
    # Helpers
    # =========================

    def _sample(self, X, y):

        n = X.shape[0]

        if self.bootstrap:
            idxs = np.random.choice(n, n, replace=True)
        else:
            idxs = np.arange(n)

        return X[idxs], y[idxs]

    def _get_seed(self, i):

        if self.random_state is None:
            return None

        return self.random_state + i
