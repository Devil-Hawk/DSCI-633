import pandas as pd
import numpy as np
from collections import Counter


class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array, or pd.Series, dependent variables, int or str
        # list of classes for this model
        self.classes_ = list(set(list(y)))                              #Extracting class labels from y and storing it in self.classes
        # for calculation of P(y)
        self.P_y = Counter(y)                                           #Counting labels for each class(as taught in class today)
        # self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi is the feature value, yj is a specific class label
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        self.P = {}
        for label in self.classes_:                                     #initializing a dictionary
            self.P[label] = {}
        class_index = 0
        while class_index < len(self.classes_):                         #This loop goes through each class label in self.classes
            label = self.classes_[class_index]
            label_mask = (y == label)
            self.P_y[label] = (np.sum(label_mask) + self.alpha) / (len(y) + len(self.classes_) * self.alpha)            #Calculating probability 'P(yj) for label
            feature_index = 0

            while feature_index < len(X.columns):                       #loop for each feature in dataframe x
                feature = X.columns[feature_index]
                self.P[label][feature] = {}                             #empty disctionary for probabilities
                value_index = 0

                while value_index < len(X[feature].unique()):           #loop for each feature
                    value = X[feature].unique()[value_index]            #goes through values in feature
                    feature_mask = (X[feature] == value)
                    self.P[label][feature][value] = (np.sum(label_mask & feature_mask) + self.alpha) / (
                                np.sum(label_mask) + len(X[feature].unique()) * self.alpha)             #calcultaing probabilities and storing it for the class label


                    value_index += 1                                        #increament for value index
                feature_index += 1                                          #increament for feature
            class_index += 1                                                #increment for class


    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # P(x|yj) = P(x1|yj)P(x2|yj)...P(xk|yj) = self.P[yj][X1][x1]*self.P[yj][X2][x2]*...*self.P[yj][Xk][xk]
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for feature in X.columns:
                p *= X[feature].apply(
                    lambda value: self.P[label][feature][value] if value in self.P[label][feature] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # Hint: predicted class is the class with highest prediction probability (from self.predict_proba)
        probs = self.predict_proba(X)                                                   #Uses predict_proba to calculate probability
        predictions = probs.idxmax(axis=1)                                              #Returns maxiumum value of each row in dataframe as taught in class today
        return predictions
