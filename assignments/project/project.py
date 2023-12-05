import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import numpy as np
from gensim.parsing.preprocessing import STOPWORDS

class my_model():
    def __init__(self):
        # Initializing variables for classifier and preprocessor
        self.clf = None
        self.preprocessor = None
        self.stopWords = STOPWORDS

    def fit(self, X, y):
        # do not exceed 29 mins
        # Concatenate relevant features into a single string
        features = X["description"] + ' ' + X["requirements"] + ' ' \
                   + X["telecommuting"].astype(str) + ' ' + X["has_company_logo"].astype(str) + ' ' + X["has_questions"].astype(str)

        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True, smooth_idf=True,
                                            ngram_range=(1, 5))
        XX = self.preprocessor.fit_transform(features)

        # Initialize and fit the TfidfVectorizer to transform text features
        sgd = SGDClassifier(class_weight="balanced", shuffle=True, random_state=20, warm_start=True)
        # Initialize and fit the SGDClassifier using RandomizedSearchCV for hyperparameter tuning
        param_dist = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': np.logspace(-6, 2, 200),
            'max_iter': np.arange(500, 5000, 500),
        }

        self.clf = RandomizedSearchCV(sgd, param_distributions=param_dist, n_iter=50, cv=5, scoring='f1', n_jobs=-1)
        self.clf.fit(XX, y)

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        # Concatenate relevant features in the test data into a single string
        features = X["description"] + ' ' + X["requirements"] + ' ' + X["telecommuting"].astype(str) \
                   + ' ' + X["has_company_logo"].astype(str) + ' ' + X["has_questions"].astype(str)

        XX = self.preprocessor.transform(features)
        predictions = self.clf.predict(XX)
        return predictions

    def calculate_f1_score(self, X, y):
        features = X["description"] + ' ' + X["requirements"] + ' ' + X["telecommuting"].astype(str) + ' ' + X["has_company_logo"].astype(str) + ' ' + X["has_questions"].astype(str)
        XX = self.preprocessor.transform(features)
        predictions = self.clf.predict(XX)
        f1_fraudulent = f1_score(y, predictions, pos_label=1)

        print("F1 Score for Fraudulent Class (label 1):", f1_fraudulent)

        return f1_fraudulent
