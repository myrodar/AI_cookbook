import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, nb_iter=1000):
        self.learning_rate = learning_rate
        self.nb_iter = nb_iter
        self.weights = None
        self.bias = 0.0

    def step_func(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        training_output = np.where(y == 0, 0, 1)


        for _ in range(self.nb_iter):
            for idx, x_i in enumerate(X):
                linear_output= np.dot(x_i, self.weights) + self.bias
                y_predicted = self.step_func(linear_output)

                update = self.learning_rate * (training_output[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Perceptron is not fitted yet. Call fit(X, y) before predict().")
        linear_output = np.dot(X, self.weights) + self.bias
        y_predict = self.step_func(linear_output)
        return y_predict

    def decision_boundary(self, x_vals: np.ndarray) -> np.ndarray:
        """Return y values of the decision boundary for 2D inputs.
        Uses w1*x + w2*y + b = 0  =>  y = -(w1*x + b)/w2
        """
        if self.weights is None:
            raise ValueError("Perceptron is not fitted yet. Call fit(X, y) before using decision_boundary().")
        if self.weights.shape[0] != 2:
            raise ValueError("decision_boundary is only defined for 2D feature vectors.")
        w1, w2 = self.weights
        return -(w1 * x_vals + self.bias) / (w2 + 1e-12)