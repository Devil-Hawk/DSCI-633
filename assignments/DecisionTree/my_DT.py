import pandas as pd
import numpy as np
from collections import Counter

class my_DT:

    def __init__(self, criterion="gini", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        #   Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)

    def impurity(self, labels):
        # Calculate impurity (unweighted)
        # Input is a list (or np.array) of labels
        # Output impurity score
        stats = Counter(labels)
        N = len(labels)
        if self.criterion == "gini":
            impurity = 1.0 - sum((count / N) ** 2 for count in stats.values())
        elif self.criterion == "entropy":
            impurity = -sum((count / N) * np.log2(count / N) for count in stats.values())
        else:
            raise Exception("Unknown criterion.")

        return impurity

    def find_best_split(self, pop, X, labels):
        # Find the best split
        # Inputs:
        #   pop:    indices of data in the node
        #   X:      independent variables of training data
        #   labels: dependent variables of training data
        # Output: tuple(best feature to split, weighted impurity score of best split, splitting point of the feature, [indices of data in left node, indices of data in right node], [weighted impurity score of left node, weighted impurity score of right node])
        ######################
        # Find the best feature and split value to split the data
        best_feature = None
        best_impurity = 1.0
        best_split_value = None
        best_left_indices = []
        best_right_indices = []
        best_left_impurity = None
        best_right_impurity = None

        for feature in X.columns:
            values = np.array(X[feature])
            unique_values = np.unique(values)

            for split_value in unique_values:
                left_indices = pop[values[pop] < split_value]
                right_indices = pop[values[pop] >= split_value]
                if len(left_indices) < self.min_samples_split or len(right_indices) < self.min_samples_split:
                    continue

                left_impurity = self.impurity(labels[left_indices])
                right_impurity = self.impurity(labels[right_indices])
                weighted_impurity = (len(left_indices) * left_impurity + len(right_indices) * right_impurity) / len(pop)

                if weighted_impurity < best_impurity:
                    best_feature = feature
                    best_impurity = weighted_impurity
                    best_split_value = split_value
                    best_left_indices = left_indices
                    best_right_indices = right_indices
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity

        if best_feature is not None:
            return best_feature, best_impurity, best_split_value, [best_left_indices, best_right_indices], [
                best_left_impurity, best_right_impurity]
        else:
            return None

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        # Build the decision tree by recursively splitting nodes
        self.classes_ = list(set(y))
        labels = np.array(y)
        N = len(y)
        ##### A Binary Tree structure implemented in the form of dictionary #####
        # 0 is the root node
        # node i have two childen: left = i*2+1, right = i*2+2
        # self.tree[i] = tuple(feature to split on, value of the splitting point) if it is not a leaf
        #              = Counter(labels of the training data in this leaf) if it is a leaf node

        self.tree = {}
        # population keeps the indices of data points in each node
        population = {0: np.array(range(N))}
        # impurity stores the weighted impurity scores for each node (# data in node * unweighted impurity).
        # NOTE: for simplicity reason we do not divide weighted impurity score by N here.
        impurity = {0: self.impurity(labels[population[0]]) * N}
        #########################################################################
        level = 0
        nodes = [0]
        while level < self.max_depth and nodes:
            # Breadth-first search to split nodes
            next_nodes = []
            for node in nodes:
                current_pop = population[node]
                current_impure = impurity[node]
                if len(current_pop) < self.min_samples_split or current_impure == 0 or level + 1 == self.max_depth:
                    # The node is a leaf node
                    # If stopping criteria are met, assign leaf node with class counts
                    self.tree[node] = Counter(labels[current_pop])
                else:
                    best_feature = self.find_best_split(current_pop, X, labels)
                    if best_feature and (current_impure - best_feature[1]) > self.min_impurity_decrease * N:
                        #Split the node
                        # If a valid split is found, add it to the tree and continue to next level
                        self.tree[node] = (best_feature[0], best_feature[2])
                        next_nodes.extend([node * 2 + 1, node * 2 + 2])
                        population[node * 2 + 1] = best_feature[3][0]
                        population[node * 2 + 2] = best_feature[3][1]
                        impurity[node * 2 + 1] = best_feature[4][0]
                        impurity[node * 2 + 2] = best_feature[4][1]
                    else:
                        # The node is a leaf node
                        # If no valid split is found, assign leaf node with class counts
                        self.tree[node] = Counter(labels[current_pop])
            nodes = next_nodes
            level += 1

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # Make predictions by traversing the decision tree
        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    label = list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
                    predictions.append(label)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        return predictions

    def predict_proba(self, X):
        # Make probability predictions by traversing the decision tree
        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    prob = {c: self.tree[node][c] / sum(self.tree[node].values()) for c in self.classes_}
                    predictions.append(prob)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        probs = pd.DataFrame(predictions, columns=self.classes_)
        return probs
