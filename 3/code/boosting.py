import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def gradient_l2(train_target, train_predict):
    """
    compute g_t in 3.5.2
    """
    return -train_target + train_predict


def gradient_logistic(train_target, train_predict):
    """
    compute g_t in 3.5.3
    """
    # Your code goes here  (~3 lines)
    # TODO 3.5.6
    sigmoid = 1 / (1 + np.exp(train_target * train_predict))
    return -train_target * sigmoid


class GradientBoosting:
    def __init__(self, T, gradient_func, learning_rate=0.1, min_sample=5, max_depth=3):
        '''
        Initialize gradient boosting class

        :param T: number of rounds of gradient boosting
        :gradient_func: function used for computing gradient
        :param learning_rate: step size of gradient descent
        '''
        self.T = T
        self.gradient_func = gradient_func
        self.learning_rate = learning_rate
        self.min_sample = min_sample
        self.max_depth = max_depth

    def fit(self, train_data, train_target):
        """
        Fit gradient boosting model
        :param train_data: x
        :param train_target: y
        :return:
        """
        ft = np.zeros(train_data.shape[0])  # f_t(x)
        train_target = train_target.squeeze()
        self.ht = []  # sequence of h_t(x)
        for t in range(self.T):
            rgs = DecisionTreeRegressor(min_samples_split=self.min_sample, max_depth=self.max_depth)
            # Your code goes here (~4 lines)
            # TODO 3.5.4
            gradient = self.gradient_func(train_target, ft)
            rgs.fit(train_data, -gradient)
            self.ht.append(rgs)
            ft += self.learning_rate * rgs.predict(train_data)

        return self

    def predict(self, test_data):
        # Your code goes here (~6 lines)
        # TODO 3.5.5
        ft = np.zeros(test_data.shape[0])
        for rgs in self.ht:
            ft += self.learning_rate * rgs.predict(test_data)
        return ft


def training_boosting_classifier():
    data_train = np.loadtxt('data/cls_train.txt')
    data_test = np.loadtxt('data/cls_test.txt')
    x_train, y_train = data_train[:, 0: 2], data_train[:, 2].reshape(-1, 1)
    x_test, y_test = data_test[:, 0: 2], data_test[:, 2].reshape(-1, 1)

    # Change target to 0-1 label
    y_train_label = np.array(list(map(lambda x: 1 if x > 0 else 0, y_train))).reshape(-1, 1)

    # Plotting decision regions
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, i, tt in zip(product([0, 1], [0, 1, 2]),
                          [1, 5, 10, 20, 50, 100],
                          ['n_estimator = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):
        gbt = GradientBoosting(T=i, gradient_func=gradient_l2, max_depth=2, learning_rate=0.1)
        gbt.fit(x_train, y_train)

        Z = np.sign(gbt.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(x_train[:, 0], x_train[:, 1], c=y_train_label, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
    plt.savefig('output/GBM_l2.pdf')

    plt.clf()

    # Plotting decision regions
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, i, tt in zip(product([0, 1], [0, 1, 2]),
                          [1, 5, 10, 20, 50, 100],
                          ['T = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):
        gbt = GradientBoosting(T=i, gradient_func=gradient_logistic, max_depth=3,
                               learning_rate=0.1)
        gbt.fit(x_train, y_train)

        Z = np.sign(gbt.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(x_train[:, 0], x_train[:, 1], c=y_train_label, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
    plt.savefig('output/GBM_logistic.pdf')


def training_boosting_regression():
    data_krr_train = np.loadtxt('data/reg_train.txt')
    data_krr_test = np.loadtxt('data/reg_test.txt')
    x_krr_train, y_krr_train = data_krr_train[:, 0].reshape(-1, 1), data_krr_train[:, 1].reshape(-1, 1)
    x_krr_test, y_krr_test = data_krr_test[:, 0].reshape(-1, 1), data_krr_test[:, 1].reshape(-1, 1)

    plot_size = 0.001
    x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))

    for idx, i, tt in zip(product([0, 1], [0, 1, 2]),
                          [1, 5, 10, 20, 50, 100],
                          ['T = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):
        gbm_1d = GradientBoosting(T=i, gradient_func=gradient_l2, max_depth=2)
        gbm_1d.fit(x_krr_train, y_krr_train)

        y_range_predict = gbm_1d.predict(x_range)

        axarr[idx[0], idx[1]].plot(x_range, y_range_predict, color='r')
        axarr[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
        axarr[idx[0], idx[1]].set_xlim(0, 1)

    plt.savefig("output/GBM_regression.pdf")


if __name__ == '__main__':
    training_boosting_classifier()
    training_boosting_regression()
