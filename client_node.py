import numpy as np
from sklearn.linear_model import LogisticRegression

class ClientNode:
    def __init__(self, node_id, is_malicious=False):
        self.node_id = node_id
        self.is_malicious = is_malicious
        self.model = LogisticRegression()

    def generate_data(self, n=200):
        X = np.random.randn(n, 10)
        y = np.random.randint(0, 2, size=n)
        if self.is_malicious:
            y = 1 - y  # flip labels to poison the model
        return X, y

    def train(self):
        X, y = self.generate_data()
        self.model.fit(X, y)
        return self.model.coef_.flatten()  # return model weights
