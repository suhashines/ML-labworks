import numpy as np
from models.base_tree import BaseTree


class DecisionTree(BaseTree):
    """
    Custom Decision Tree classifier.
    Uses all features for split selection (no feature sub-sampling).
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        random_state=None
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state
        )

    def _get_feature_indices(self, num_features):
        """
        For a standard Decision Tree, consider all features.
        """
        return np.arange(num_features)
