import numpy as np

class KNN(object):
    
    def __init__(self, k):
        self.k = k
    
    def squared_distances(self, a, b):
        
        sum_a = np.sum(a ** 2, -1).reshape(-1, 1)
        sum_b = np.sum(b ** 2, -1).reshape(1, -1)
        distances = np.matmul(a, b.transpose())
        distances *= -2
        distances += sum_a
        distances += sum_b
        return np.clip(distances, 0., None)


class KNNRegressor(KNN):
    
    def __init__(self, k):
        super().__init__(k)
    
    def fit(self, X, y):
        self.X, self.y = X, y
        
    def predict(self, X_test):
        
        dm = self.squared_distances(X_test, self.X)
        y_hat = np.array([self.y[np.argsort(dm[i])[:self.k]].mean() for i in range(len(dm))])
        return y_hat
    
    def score(self, X_test, y_test):
        
        assert len(X_test) == len(y_test)
        y_hat = self.predict(X_test)
        return 1. - np.var(y_test - y_hat) / max(np.var(y_test), 1e-12)


class KNNClassifier(KNN):
    
    def __init__(self, k):
        super().__init__(k)
    
    def fit(self, X, y):
        self.X, self.y = X, y
        
    def predict(self, X_test):
        
        dm = self.squared_distances(X_test, self.X)
        top_k = np.argsort(dm, -1)[:, :self.k]
        labels = np.arange(len(set(self.y))).reshape(1, -1, 1)
        y_hat = np.argmax(np.sum(
            np.equal(np.expand_dims(self.y[top_k], 1), labels),
            -1), -1)
        return y_hat
    
    def score(self, X_test, y_test):
        
        assert len(X_test) == len(y_test)
        y_hat = self.predict(X_test)
        return np.mean(np.equal(y_test, y_hat))