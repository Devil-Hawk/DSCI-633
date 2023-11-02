import pandas as pd
import numpy as np

class my_Logistic:

    def __init__(self, learning_rate=0.1, batch_size=10, max_iter=100, shuffle=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        #fitting the logistic regression model to training data
        data = X.to_numpy()
        d = data.shape[1]
        self.w = np.array([0.0] * d)
        self.w0 = 0.0
        n = len(y)
        for epoch in range(self.max_iter):
            if self.shuffle:
                data, y = self.shuffle_data(data, y)
            batches = self.generate_batches(n)
            for batch in batches:
                X_train = data[batch]
                y_train = y[batch]
                self.w, self.w0 = self.sgd(X_train, y_train, self.w, self.w0)

    def generate_batches(self, n):
        #Generating batches of data for stocastic gradient descent
        if self.batch_size >= n:
            return [list(range(n))]
        num_batches = n // self.batch_size
        remainder = n % self.batch_size
        batches = [list(range(i * self.batch_size, (i + 1) * self.batch_size)) for i in range(num_batches)]
        if remainder > 0:
            batches.append(list(range(num_batches * self.batch_size, n)))
        return batches

    def sgd(self, X, y, w, w0):
        #performing sgd for a batch
        wx = np.dot(X, w) + w0
        fx = 1.0 / (1 + np.exp(-wx))
        gradient_w = np.dot(X.T, fx - y) / X.shape[0]
        gradient_w0 = np.sum(fx - y) / X.shape[0]
        w -= self.learning_rate * gradient_w
        w0 -= self.learning_rate * gradient_w0
        return w, w0

    def shuffle_data(self, X, y):
        #shuffling data
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def predict_proba(self, X):
        #classs probabilities for input
        data = X.to_numpy()
        wx = np.dot(data, self.w) + self.w0
        fx = 1.0 / (1 + np.exp(-wx))
        return fx

    def predict(self, X):
        #Predict binary class labels
        probs = self.predict_proba(X)
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]
        return predictions