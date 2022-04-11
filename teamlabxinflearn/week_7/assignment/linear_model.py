import numpy as np

class LinearRegression(object):
    def __init__(self, fit_intercept=True):#, copy_X=True):
        self.fit_intercept = fit_intercept
        
        self.copy_X = None
        self._coef = None
        self._intercept = None
        self._new_X = None
        self._weights = None

    def fit(self, X, y):
        self._new_X = np.array(X)
        if self.fit_intercept == True:
            self._new_X = np.append(np.ones_like(self._new_X),self._new_X, axis = 1)
        self._weights = np.linalg.inv(self._new_X.T.dot(self._new_X)).dot(self._new_X.T).dot(y) # 해당 값 역행렬
        self._coef = self._weights[1:]
        self._intercept = self._weights[0]

    def predict(self, X):
        self.copy_X = np.array(X)
        if self.fit_intercept == True:
            self.copy_X = np.append(np.ones_like(self.copy_X),self.copy_X, axis = 1)
        y_hat = self.copy_X.dot(self._weights)
        return y_hat

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept