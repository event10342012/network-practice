import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


def sign(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(x > threshold, 1, -1)


class Perceptron(object):
    def __init__(self, epochs: int = 10, threshold: float = 0, learning_rate: float = 0.1):
        self.epochs = epochs
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.scores = []

    def _initialize_weight(self, d: int):
        self.W = np.random.rand(d)
        self.b = np.random.rand(1, )

    def _forward(self, X):
        return np.dot(X, self.W) + self.b

    def fit(self, X: np.ndarray, Y: np.ndarray, plot=False):
        n, d = X.shape
        self._initialize_weight(d)

        # epochs
        for _ in range(self.epochs):
            # iter all samples
            i = 0
            for x, y in zip(X, Y):
                # compute the Z = XW + b
                z = np.dot(x, self.W) + self.b

                # activation
                y_ = sign(z, self.threshold)

                # cost function 1/n * sigma |y_ - y| for i = 1, 2, ... n
                if not np.equal(y_, y):
                    i += 1
                    self.W += self.learning_rate * y * x
                    self.b += self.learning_rate * y

                    if plot:
                        index_1 = np.where(Y == 1)[0]
                        index_2 = np.where(Y == -1)[0]
                        x1 = X[index_1]
                        x2 = X[index_2]
                        plt.scatter(x1[:, 0], x1[:, 1], color='r')
                        plt.scatter(x2[:, 0], x2[:, 1], color='b')
                        self.plot_decision_boundary(x)
                        print('Iteration:', i, 'Accuracy:', self.score(X, Y))
                        plt.show()

            self.scores.append(self.score(X, Y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_ = np.dot(X, self.W) + self.b
        return sign(y_, self.threshold)

    def score(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Y_ = self.predict(X)
        return (Y == Y_).mean()

    def plot_decision_boundary(self, x: np.ndarray):
        x_vector = np.linspace(4, 7.2)
        y_vector = -(self.W[0] / self.W[1]) * x_vector - (self.b / self.W[1])
        plt.scatter(x[0], x[1], color='g')
        plt.plot(x_vector, y_vector, color='black')


class LinearRegression(Perceptron):
    def __init__(self, epochs: int = 10, learning_rate: float = 0.05):
        super().__init__(epochs=epochs, learning_rate=learning_rate)

    def fit(self, X: np.ndarray, Y: np.ndarray, plot=False):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class LogisticRegression(object):
    def __init__(self):
        pass

    def backward(self):
        pass

    def back_propagation(self):
        pass


if __name__ == '__main__':
    # load dataseet
    iris = load_iris()
    x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.DataFrame(iris['target'], columns=['target'])
    iris_df = pd.concat([x, y], axis=1)
    iris_df = shuffle(iris_df)

    train = iris_df[iris_df['target'] != 2]
    train['target'][train['target'] == 0] = -1
    X_train = train.iloc[:, :2].values
    y_train = train.iloc[:, -1].values

    model = Perceptron(1)
    model.fit(X_train, y_train, plot=True)
    model.score(X_train, y_train)
