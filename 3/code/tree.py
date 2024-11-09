import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTree(BaseEstimator):

    def __init__(self, split_loss_function, leaf_value_estimator,
                 depth=0, min_sample=5, max_depth=10):
        """
        Initialize the decision tree

        :param split_loss_function: method for splitting node
        :param leaf_value_estimator: method for estimating leaf value
        :param depth: depth indicator, default value is 0, representing root node
        :param min_sample: an internal node can be splitted only if it contains points more than min_smaple
        :param max_depth: restriction of tree depth.
        """
        self.split_loss_function = split_loss_function
        self.leaf_value_estimator = leaf_value_estimator
        self.depth = depth
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.split_id = None
        self.split_value = None
        self.left = None
        self.right = None
        self.is_leaf = None
        self.leaf_value = None

    def fit(self, X, y=None):
        """
        This should fit the tree classifier by setting the values self.is_leaf,
        self.split_id (the index of the feature we want ot split on, if we're splitting),
        self.split_value (the corresponding value of that feature where the split is),
        and self.leaf_value, which is the prediction value if the tree is a leaf node.  If we
        are splitting the node, we should also init self.left and self.right to be DecisionTree
        objects corresponding to the left and right subtrees. These subtrees should be fit on
        the data that fall to the left and right, respectively, of self.split_value.
        This is a recurisive tree building procedure.

        :param X: a numpy array of training data, shape = (n, m)
        :param y: a numpy array of labels, shape = (n, 1)

        :return self
        """
        # If depth is max depth turn into leaf
        if self.depth == self.max_depth:
            self.is_leaf = True
            self.leaf_value = self.leaf_value_estimator(y)
            return self

        # If reach minimun sample size turn into leaf
        if len(y) <= self.min_sample:
            self.is_leaf = True
            self.leaf_value = self.leaf_value_estimator(y)
            return self

        # If not is_leaf, i.e in the node, we should create left and right subtree
        # But First we need to decide the self.split_id and self.split_value that minimize loss
        # Compare with constant prediction of all X
        best_split_value = None
        best_split_id = None
        best_loss = self.split_loss_function(y)
        best_left_X = None
        best_right_X = None
        best_left_y = None
        best_right_y = None
        # Concatenate y into X for sorting together
        X = np.concatenate([X, y], 1)
        for i in range(X.shape[1] - 1):
            # Note: The last column of X is y now
            X = np.array(sorted(X, key=lambda x: x[i]))
            for split_pos in range(len(X) - 1):
                # :split_pos+1 will include the split_pos data in left_X
                left_X = X[:split_pos + 1, :-1]
                right_X = X[split_pos + 1:, :-1]
                # you need left_y to be in (n,1) i.e (-1,1) dimension
                left_y = X[:split_pos + 1, -1].reshape(-1, 1)
                right_y = X[split_pos + 1:, -1].reshape(-1, 1)
                left_loss = len(left_y) * self.split_loss_function(left_y) / len(y)
                right_loss = len(right_y) * self.split_loss_function(right_y) / len(y)
                # If any choice of splitting feature and splitting position results in better loss
                # record following information and discard the old one
                if ((left_loss + right_loss) < best_loss):
                    best_split_value = X[split_pos, i]
                    best_split_id = i
                    best_loss = left_loss + right_loss
                    best_left_X = left_X
                    best_right_X = right_X
                    best_left_y = left_y
                    best_right_y = right_y

        # Condition when you have a split position that results in better loss
        # Your code goes here (~10 lines)
        # TODO 2.3.3
        if best_split_id != None:
            # build child trees and set spliting info
            self.split_id = best_split_id
            self.split_value = best_split_value
            self.left = DecisionTree(self.split_loss_function, self.leaf_value_estimator, self.depth+1, self.min_sample, max_depth=self.max_depth)
            self.right = DecisionTree(self.split_loss_function, self.leaf_value_estimator, self.depth+1, self.min_sample, max_depth=self.max_depth)
            self.left.fit(best_left_X, best_left_y)
            self.right.fit(best_right_X, best_right_y)
        else:
            # set value
            self.is_leaf = True
            self.leaf_value = self.leaf_value_estimator(y)

        return self

    def predict_instance(self, instance):
        """
        Predict label by decision tree

        :param instance: a numpy array with new data, shape (1, m)

        :return whatever is returned by leaf_value_estimator for leaf containing instance
        """
        if self.is_leaf:
            return self.leaf_value
        if instance[self.split_id] <= self.split_value:
            return self.left.predict_instance(instance)
        else:
            return self.right.predict_instance(instance)


def compute_entropy(label_array):
    """
    Calulate the entropy of given label list

    :param label_array: a numpy array of labels shape = (n, 1)
    :return entropy: entropy value
    """
    # Your code goes here (~6 lines)
    # TODO 2.3.1
    _, label_counts = np.unique(label_array, return_counts=True)
    prob = label_counts / len(label_array)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def compute_gini(label_array):
    """
    Calulate the gini index of label list

    :param label_array: a numpy array of labels shape = (n, 1)
    :return gini: gini index value
    """
    # Your code goes here (~6 lines)
    # TODO 2.3.2
    _, label_counts = np.unique(label_array, return_counts=True)
    prob = label_counts / len(label_array)
    gini = 1 - np.sum(prob ** 2)
    return gini


def most_common_label(y):
    """
    Find most common label
    """
    label_cnt = Counter(y.reshape(len(y)))
    label = label_cnt.most_common(1)[0][0]
    return label


class ClassificationTree(BaseEstimator, ClassifierMixin):

    loss_function_dict = {
        'entropy': compute_entropy,
        'gini': compute_gini
    }

    def __init__(self, loss_function='entropy', min_sample=5, max_depth=10):
        """
        :param loss_function(str): loss function for splitting internal node
        """

        self.tree = DecisionTree(self.loss_function_dict[loss_function],
                                 most_common_label,
                                 0, min_sample, max_depth)

    def fit(self, X, y=None):
        self.tree.fit(X, y)
        return self

    def predict_instance(self, instance):
        value = self.tree.predict_instance(instance)
        return value


# Regression Tree Specific Code
def mean_absolute_deviation_around_median(y):
    """
    Calulate the mean absolute deviation around the median of a given target list

    :param y: a numpy array of targets shape = (n, 1)
    :return mae
    """
    # Your code goes here  (~3 lines)
    # TODO 2.3.4
    mae = np.mean(np.abs(y - np.median(y)))
    return mae


class RegressionTree:
    """
    :attribute loss_function_dict: dictionary containing the loss functions used for splitting
    :attribute estimator_dict: dictionary containing the estimation functions used in leaf nodes
    """

    loss_function_dict = {
        'mse': np.var,
        'mae': mean_absolute_deviation_around_median
    }

    estimator_dict = {
        'mean': np.mean,
        'median': np.median
    }

    def __init__(self, loss_function='mse', estimator='mean', min_sample=5, max_depth=10):
        """
        Initialize RegressionTree
        :param loss_function(str): loss function used for splitting internal nodes
        :param estimator(str): value estimator of internal node
        """

        self.tree = DecisionTree(self.loss_function_dict[loss_function],
                                 self.estimator_dict[estimator],
                                 0, min_sample, max_depth)

    def fit(self, X, y=None):
        self.tree.fit(X, y)
        return self

    def predict_instance(self, instance):
        value = self.tree.predict_instance(instance)
        return value


def training_classification_tree():
    # Load Data
    data_train = np.loadtxt('data/cls_train.txt')
    data_test = np.loadtxt('data/cls_test.txt')
    x_train, y_train = data_train[:, 0: 2], data_train[:, 2].reshape(-1, 1)
    x_test, y_test = data_test[:, 0: 2], data_test[:, 2].reshape(-1, 1)

    # Change target to 0-1 label
    y_train_label = np.array(list(map(lambda x: 1 if x > 0 else 0, y_train))).reshape(-1, 1)

    # Training classifiers with different depth
    clfs = []
    for depth in range(1, 7):
        clf = ClassificationTree(max_depth=depth)
        clf.fit(x_train, y_train_label)
        clfs.append(clf)

    # Plotting decision regions
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1, 2]),
                            clfs,
                            ['Depth = {}'.format(n) for n in range(1, 7)]):
        Z = np.array([clf.predict_instance(x) for x in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(x_train[:, 0], x_train[:, 1], c=y_train_label, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
    plt.savefig('output/DT_entropy.pdf')


def training_regression_tree():
    data_krr_train = np.loadtxt('data/reg_train.txt')
    data_krr_test = np.loadtxt('data/reg_test.txt')
    x_krr_train, y_krr_train = data_krr_train[:, 0].reshape(-1, 1), data_krr_train[:, 1].reshape(-1, 1)
    x_krr_test, y_krr_test = data_krr_test[:, 0].reshape(-1, 1), data_krr_test[:, 1].reshape(-1, 1)

    # Training regression trees with different depth
    regs = []
    for depth in range(1, 7):
        reg = RegressionTree(max_depth=depth)
        reg.fit(x_krr_train, y_krr_train)
        regs.append(reg)

    plot_size = 0.001
    x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))

    for idx, reg, tt in zip(product([0, 1], [0, 1, 2]),
                            regs,
                            ['Depth = {}'.format(n) for n in range(1, 7)]):
        y_range_predict = np.array([reg.predict_instance(x) for x in x_range]).reshape(-1, 1)

        axarr[idx[0], idx[1]].plot(x_range, y_range_predict, color='r')
        axarr[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
        axarr[idx[0], idx[1]].set_xlim(0, 1)
    plt.savefig('output/DT_regression.pdf')


if __name__ == '__main__':
    training_classification_tree()
    training_regression_tree()
