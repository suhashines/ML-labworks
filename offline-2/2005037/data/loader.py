import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """
    Dataset loader utility for classification datasets.
    Supports Iris and Wine datasets.
    """

    def __init__(self, test_size=0.2, random_state=42, stratify=True):
        """
        Parameters
        ----------
        test_size : float
            Proportion of dataset to include in the test split
        random_state : int
            Seed for reproducibility
        stratify : bool
            Whether to stratify splits by class labels
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def load(self, name):
        """
        Load dataset by name.

        Parameters
        ----------
        name : str
            Dataset name ('iris' or 'wine')

        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        """
        name = name.lower()

        if name == "iris":
            data = load_iris()
        elif name == "wine":
            data = load_wine()
        else:
            raise ValueError("Unsupported dataset. Choose from ['iris', 'wine'].")

        X = data.data
        y = data.target

        return X, y

    def train_test_split(self, X, y):
        """
        Perform train-test split on dataset.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        stratify_labels = y if self.stratify else None

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )

    def load_and_split(self, name):
        """
        Convenience method to load a dataset and split it.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        X, y = self.load(name)
        return self.train_test_split(X, y)

# test the DatasetLoader
if __name__ == "__main__":
    loader = DatasetLoader(test_size=0.3, random_state=1, stratify=True)
    X_train, X_test, y_train, y_test = loader.load_and_split("iris")

    print("Iris Dataset:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:", np.bincount(y_train))
    print("y_test distribution:", np.bincount(y_test))
    X_train, X_test, y_train, y_test = loader.load_and_split("wine")
    print("\nWine Dataset:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:", np.bincount(y_train))
    