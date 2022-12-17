import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
import time


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None, random_state=0,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.max_depth = max_depth
        self.trees_parameters = trees_parameters
        self.obj = []
        self.ind = []

    def fit(self, X, y, X_val=None, y_val=None, ret_train=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        np.random.seed(self.random_state)
        score_train = []
        score = []
        time_work = []
        time_start = time.time()
        for i in range(self.n_estimators):
            self.ind.append(np.arange(X.shape[1]))
            np.random.shuffle(self.ind[i])
            if self.feature_subsample_size is None:
                self.ind[i] = self.ind[i][:int(X.shape[1] / 3)]
            else:
                self.ind[i] = self.ind[i][:int(self.feature_subsample_size * X.shape[1])]
            d = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state,
                                      **self.trees_parameters)
            self.obj.append(d.fit(X[:, self.ind[i]], y))
            time_end = time.time()
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                score.append(np.sqrt(np.sum(y_pred - y_val) ** 2))
                time_work.append((time_end - time_start) * 1000)
            if ret_train:
                score_train.append(np.sqrt(np.sum(self.predict(X) - y) ** 2))
        if X_val is not None and y_val is not None:
            if ret_train:
                return np.array(score), np.array(time_work), np.array(score_train)
            else:
                return np.array(score), np.array(time_work)
        if ret_train:
            return np.array(score_train)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pr = np.zeros((len(self.obj), X.shape[0]))
        for i in range(len(self.obj)):
            pr[i] = self.obj[i].predict(X[:, self.ind[i]])
        return np.mean(pr, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None, random_state=0,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees_parameters = trees_parameters
        self.obj = []
        self.b_0 = 0
        self.g = []
        self.ind = []

    def fit(self, X, y, X_val=None, y_val=None, ret_train=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(self.random_state)
        score = []
        score_train = []
        time_work = []
        self.b_0 = np.mean(y)
        s = 0
        time_start = time.time()
        for i in range(self.n_estimators):
            if i == 0:
                s = y - self.b_0
            else:
                s = s - self.learning_rate * self.g[i-1] * self.obj[i-1].predict(X[:, self.ind[i-1]])
            self.ind.append(np.arange(X.shape[1]))
            np.random.shuffle(self.ind[i])
            if self.feature_subsample_size is None:
                self.ind[i] = self.ind[i][:int(X.shape[1] / 3)]
            else:
                self.ind[i] = self.ind[i][:int(self.feature_subsample_size * X.shape[1])]

            d = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state, **self.trees_parameters)
            self.obj.append(d.fit(X[:, self.ind[i]], s))
            self.g.append(minimize_scalar(lambda x: np.sum((x * self.learning_rate *
                                                            self.obj[i].predict(X[:, self.ind[i]]) - s) ** 2)).x)
            time_end = time.time()
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                score.append(np.sqrt(np.sum(y_pred - y_val) ** 2))
                time_work.append((time_end - time_start) * 1000)
            if ret_train:
                score_train.append(np.sqrt(np.sum(self.predict(X) - y) ** 2))
        if X_val is not None and y_val is not None:
            if ret_train:
                return np.array(score), np.array(time_work), np.array(score_train)
            else:
                return np.array(score), np.array(time_work)
        if ret_train:
            return np.array(score_train)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = self.b_0
        for i in range(len(self.obj)):
            res += self.g[i] * self.obj[i].predict(X[:, self.ind[i]])
        return res
#
# data = pd.read_csv('kc_house_data.csv')
# X = data.drop(['price'], axis=1)
# y = data['price']
# mod = RandomForestMSE(1)
# mod.fit(X, y)