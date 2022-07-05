from model.core import Model
import numpy as np

class LogisticRegressor(Model):
    def __init__(self):
        self.w = None
        self.b = 0.0
        self.length = 0
        self.lr = 0.01
        self.max_iter = 500
        return

    def build(self, train_X):
        self.length = train_X.shape[1]
        self.w = np.random.normal(size = (self.length))
        self.b = np.random.normal()
        return

    def sigmoid(self, z):
        return np.where(z >= 1, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    
    def loss(self, a, y):
        l = -y * np.log2(a) - (1 - y) * np.log2(1 - a)
        return  np.mean(l)

    def train_with_weights(self, train_X, train_Y, weights):
        self.build(train_X)
        for it in range(self.max_iter):
            z = train_X.dot(self.w) + self.b
            a = self.sigmoid(z)
            grad = train_X.T.dot((a - train_Y) * weights)
            self.w -= self.lr * grad
            self.b -= self.lr * np.sum((a - train_Y) * weights)
        return
    def evaluate(self, X):
        return self.sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        a = self.sigmoid(X.dot(self.w) + self.b)
        return np.where(a >= 0.5, 1, 0)