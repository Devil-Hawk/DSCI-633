import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1 - cosine(x, y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric  # Stores the type of metric in self
        self.p = p  # Only needed for minkowski

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array, or pd.Series, dependent variables, int or str
        self.classes_ = np.unique(y)                                            # Storing different classes in y using numpy
        self.X_train = X                                                        # Storing training data
        self.y_train = y
        return

    def dist(self, x):
        # Calculate distances of training data to a single input data point (distances from self.X_train to x)
        # Output np.array([distances to x])
        if self.metric == "minkowski":
            # Since p=2 which is given, we calculate the Euclidean distance only using the same formula
            distances = np.power(np.sum(np.power(np.abs(self.X_train - x), self.p), axis=1), 1 / self.p)
        elif self.metric == "euclidean":
            distances = np.sqrt(np.sum(np.power(self.X_train - x, 2), axis=1))
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == "cosine":
            # Dot product: Calculate dot product between self.X_train and x
            # Norm_x: Calculating magnitude of x
            # Norm_data: Calculating normal for each data point
            # Cosine_similarity: Calculating cosine similarity for all the data points using the formula below
            # Computing distances in the last
            dot_product = np.dot(self.X_train, x)
            norm_x = np.sqrt(np.sum(np.square(x)))
            norm_data = np.sqrt(np.sum(self.X_train ** 2, axis=1))
            cosine_similarity = dot_product / (norm_x * norm_data)
            distances = 1 - cosine_similarity
        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)  # Calculating distances between point x and all others
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        return Counter(nearest_labels)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = []

        for index, row in X.iterrows():                                             # Going through each row
            neighbors = self.k_neighbors(row)  # Finding k nearest neighbors
            most_common_label = neighbors.most_common(1)[0][0]
            predictions.append(most_common_label)                                   # Appending the most common label to the list

        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each category
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        for index, row in X.iterrows():                                             # Going through each row
            neighbors = self.k_neighbors(row)                                       # Finding k nearest neighbors
            label_counts = neighbors                                                # For the counter
            label_probs = {label: count / self.n_neighbors for label, count in label_counts.items()}  # Class probability for each label
            probs.append(label_probs)                                               # Appending to the probs list
        return pd.DataFrame(probs, columns=self.classes_)
