import numpy as np
import pandas as pd

from sklearn.base import clone


class MetaCost:
    """A procedure for making error-based classifiers cost-sensitive

    Adapted from https://github.com/Treers/MetaCost/blob/master/MetaCost.py

    .. note:: The form of the cost matrix C must be as follows:
    +---------------+----------+----------+----------+
    |  actual class |          |          |          |
    +               |          |          |          |
    |   +           | y(x)=j_1 | y(x)=j_2 | y(x)=j_3 |
    |       +       |          |          |          |
    |           +   |          |          |          |
    |predicted class|          |          |          |
    +---------------+----------+----------+----------+
    |   h(x)=j_1    |    0     |    a     |     b    |
    |   h(x)=j_2    |    c     |    0     |     d    |
    |   h(x)=j_3    |    e     |    f     |     0    |
    +---------------+----------+----------+----------+
    | C = np.array([[0, a, b],[c, 0 , d],[e, f, 0]]) |
    +------------------------------------------------+
    """

    def __init__(self, estimator, cost_matrix, n_estimators=50, n_samples=None, p=True, q=True):
        """
        Parameters
        ----------
        estimator :
            An sklearn classifier
        cost_matrix :
            The cost matrix
        n_estimators :
            The number of estimators in the ensemble
        n_samples :
            The number of samples to train each estimator
        p :
            Is True if the estimator produces class probabilities. False otherwise
        q :
            True if all samples are to be used for each example
        """

        self.estimator = estimator
        self.cost_matrix = cost_matrix
        self.n_estimators = n_estimators
        self. n_samples = n_samples
        self.p = p
        self.q = q

    def fit(self, X, y):
        """
        Parameters
        ----------
        X :
            Training set
        y :
            Target
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError('S must be a DataFrame object')

        X = X.copy()

        # reset index, helps with resampling
        X.reset_index(inplace=True, drop=True)
        y.index = X.index

        variables = list(X.columns)

        # concatenate
        S = pd.concat([X,y], axis=1)
        S.columns = variables + ['target']

        num_class = y.nunique()

        if not self.n_samples:
            self.n_samples = len(X)

        S_ = {} # list of subdatasets
        M = []  # list of models

        print('resampling data and training ensemble')
        for i in range(self.n_estimators):

            # Let S_[i] be a resample of S with self.n examples
            S_[i] = S.sample(n=self.n_samples, replace=True)

            X = S_[i][variables].values
            y = S_[i]['target'].values

            # Let M[i] = model produced by applying L to S_[i]
            model = clone(self.estimator)
            M.append(model.fit(X, y))

        print('Finished training ensemble')

        label = []
        S_array = S[variables].values
        # for each observation
        print('evaluating optimal class per observation')
        for i in range(len(S)):
            if self.q:
                # consider the predictions of all models
                M_ = M
            else:
                # consider the predictions of models which were not train on
                # this particular observation
                k_th = [k for k, v in S_.items() if i not in v.index]
                M_ = list(np.array(M)[k_th])

            if self.p:
                P_j = [model.predict_proba(S_array[[i]]) for model in M_]
            else:
                P_j = []
                vector = [0] * num_class
                for model in M_:
                    vector[model.predict(S_array[[i]])] = 1
                    P_j.append(vector)

            # Calculate P(j|x)
            # the average probability of each class, when combining all models
            P = np.array(np.mean(P_j, 0)).T

            # Relabel:
            label.append(np.argmin(self.cost_matrix.dot(P)))
        print('Finished re-assigning labels')

        # Model produced by applying L to S with relabeled y
        print('Training model on new data')
        X_train = S[variables].values
        y_train = np.array(label)
        self.estimator.fit(X_train, y_train)
        print('Finished training model on data with new labels')
        self.y_ = pd.Series(label)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        try:
            probs = self.estimator.predict_proba(X)
        except:
            probs = None
            print('this estimator does not support predict_proba')
        return probs