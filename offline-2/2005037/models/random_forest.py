import numpy as np
from models.decision_tree import DecisionTree


class RandomForest:
    """
    Custom Random Forest classifier.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
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

    # =========================
    # Training
    # =========================

    def fit(self, X, y):
        self.trees = []
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]

        for i in range(self.n_estimators):

            X_sample, y_sample = self._sample(X, y)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self._get_tree_seed(i)
            )

            tree.max_features = self._get_max_features()

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # =========================
    # Prediction
    # =========================

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
    # Utilities
    # =========================

    def _sample(self, X, y):

        n_samples = X.shape[0]

        if self.bootstrap:
            idxs = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True
            )
        else:
            idxs = np.arange(n_samples)

        return X[idxs], y[idxs]

    def _get_max_features(self):

        if self.max_features == "sqrt":
            return int(np.sqrt(self.n_features_))

        if self.max_features == "log2":
            return int(np.log2(self.n_features_))

        if isinstance(self.max_features, int):
            return self.max_features

        return self.n_features_

    def _get_tree_seed(self, i):
        if self.random_state is None:
            return None

        return self.random_state + i
