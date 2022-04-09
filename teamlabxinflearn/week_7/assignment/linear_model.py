import numpy as np

class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        self._coef = None
        self._intercept = None
        self._new_X = None

    def fit(self, X, y):
        self._new_X = np.array(X)
        if self.fit_intercept == True:
            self._new_X = np.append(np.ones_like(self._new_X),self._new_X, axis = 1)
        weights = np.linalg.inv(self._new_X.T.dot(self._new_X)).dot(self._new_X.T).dot(y) # 해당 값 역행렬
        self._coef = weights[1:]
        self._intercept = weights[0]

    def predict(self, X):
        pass

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

# from matplotlib.pyplot import axis
# import numpy as np


# class LinearRegression(object):
#     def __init__(self, fit_intercept=True, copy_X = True):
#         self.fit_intercept = fit_intercept
#         self.copy_X = copy_X
        
        
#         if self.fit_intercept == True:
#             self._new_X = np.append(np.ones_like(copy_X),copy_X, axis = 1)
        
#         self._coef = None #self.fit(self._new_X, self.copy_y)[1:]
#         self._intercept = None #self.fit(self._new_X, self.copy_y)[0]

#     def fit(self, X, y):
#         return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # 해당 값 역행렬
        

#     def predict(self, X):
#         pass

#     @property
#     def coef(self):
#         return self._coef

#     @property
#     def intercept(self):
#         return self._intercept
