import matplotlib.pyplot as plt
import numpy as np
import IPython
from sklearn.datasets import make_blobs, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import mean_squared_error


class Perceptron:
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], "x and y should have the same number of rows"
        self._fit(x_train, y_train)
        assert self.w.shape   == (x_train.shape[1], 1)
        return self

    def predict(self, x):
        assert x.shape[1] == self.w.shape[0]
        y_predictions = self._predict(x)
        y_predictions = np.array(y_predictions)
        assert y_predictions.shape[0] == x.shape[0], "Predictions should have the same number of rows as the input x"
        assert np.bitwise_or(y_predictions == 0, y_predictions == 1).all(), "predictions have to be 0 or 1"
        return y_predictions

    def _fit(self, x_train, y_train):
        #x_train = np.c_[np.ones(x_train.shape[0]), x_train] to add bias
        if self.w is None:
            #self.w = np.zeros((x_train.shape[1],1))
             self.w = np.random.rand(x_train.shape[1],1)
             self.w = self.w * 2 - 1 # random numbers between -1 and 1

        for i in range(self.max_iter):
            a = np.dot(self.w.T,  x_train.T)
            z = np.heaviside(a,1)
            #self.w = self.w + (self.learning_rate * np.dot((y_train - z), x_train)).T GD

            if (y_train == z).all():
                print("All  right after ", i ,"iteration")
                print(self.w)
                break

            index = i%y_train.shape[0]
            self.w = self.w + self.learning_rate * (np.dot((y_train[index]- z[0][index]),x_train[index])).reshape(-1,1)

    def _predict(self, x):
        #x = np.c_[np.ones(x.shape[0]), x]
        return np.heaviside(np.dot(self.w.T, x.T),1).T


def load_data():
    x, y = make_blobs(n_features=2, centers=2, random_state=3)
    print("x shape is:", x.shape)
    print("y shape is:", y.shape)
    print(y)
    assert np.bitwise_or(y == 0, y == 1).all()
    return x, y


def load_non_linearly_separable_data():
    """
    Generates non-linearly separable data and returns the samples and class labels
    :return:
    """
    x, y = make_gaussian_quantiles(n_features=2, n_classes=2, random_state=1)
    assert np.bitwise_or(y == 0, y == 1).all()
    return x, y


def plot_data(x, y):
    plt.figure()
    plt.title("Two linearly-separable classes", fontsize='small')
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
    #plt.show()


def plot_decision_boundary(perceptron, x, y,name):
    dim1_max, dim1_min = np.max(x[:, 0]), np.min(x[:, 0])
    dim2_max, dim2_min = np.max(x[:, 1]), np.min(x[:, 1])
    dim1_vals, dim2_vals = np.meshgrid(np.arange(dim1_min, dim1_max, 0.1),
                                       np.arange(dim2_min, dim2_max, 0.1))
    y_vals = perceptron.predict(np.c_[dim1_vals.ravel(), dim2_vals.ravel()])
    y_vals = y_vals.reshape(dim1_vals.shape)

    plt.figure()
    plt.title("Two linearly-separable classes with decision boundary" + name, fontsize='small')
    plt.contourf(dim1_vals, dim2_vals, y_vals, alpha=0.4)
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)


def main():
    x, y = load_data()
    # x, y = load_non_linearly_separable_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    learning_rate = 0.05
    n_iter = 100

    # Perceptron from sklearn
    perceptron = SkPerceptron(alpha=learning_rate, max_iter=n_iter, fit_intercept=False)
    perceptron.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, perceptron.predict(x_train))
    test_mse = mean_squared_error(y_test, perceptron.predict(x_test))
    print("Training MSE:", train_mse)
    print("Testing MSE: ", test_mse)
    plot_decision_boundary(perceptron, x, y, " SkPerceptron with eta: " + str(learning_rate) + " and max_it: " +str(n_iter))

    # Your own perceptron
    perceptron = Perceptron(learning_rate=learning_rate, max_iter=n_iter)
    perceptron.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, perceptron.predict(x_train))
    test_mse = mean_squared_error(y_test, perceptron.predict(x_test))
    print("Training MSE:", train_mse)
    print("Testing MSE: ", test_mse)
    plot_decision_boundary(perceptron, x, y, " own perceptron with eta: " + str(learning_rate) + " and max_it: " +str(n_iter))
    plt.show()


if __name__ == '__main__':
    main()
