import numpy as np


def n_size_ndarray_creation(n, dtype=np.int):
    X = np.arange(n*n, dtype=dtype).reshape(n,-1)
    return X


def zero_or_one_or_empty_ndarray(shape, type, dtype=np.int):
    if type == 0:
        X = np.zeros(shape=shape, dtype=dtype)
    elif type == 1:
        X = np.ones(shape=shape, dtype=dtype)
    else:
        X = np.empty(shape=shape, dtype=dtype)
    return X

def change_shape_of_ndarray(X, n_row):
    return X.reshape(n_row, -1) if n_row == 1 else X.flatten()


def concat_ndarray(X_1, X_2, axis):
    try:
        if X_1.ndim() == 1:
            X_1 = X_1.reshape(1,-1)
        if X_2.ndim() == 1:
            X_2 = X_2.reshape(1,-1)
        return np.concatenate(X_1, X_2, axis=axis)
    except:
        return False


def normalize_ndarray(X, axis=99, dtype=np.float32):
    # X = X.astype(dtype = dtype)
    X_row, X_col = X.shape

    if axis == 99:
        X_mean = np.mean(X)
        X_std = np.std(X)
        answer = (X - X_mean)/X_std
    if axis == 0 :
        answer = (X - np.mean(X, axis=axis))/np.std(X, axis=axis)  
    if axis == 1:
        x_mean = np.mean(X, 1).reshape(X_row, -1)
        x_std = np.std(X, 1).reshape(X_row, -1)
        answer = (X - x_mean) / x_std 
    return answer


def save_ndarray(X, filename="test.npy"):
    pass


def boolean_index(X, condition):
    return np.where(eval(str('X') + "== 3"))


def find_nearest_value(X, target_value):
    return np.argmin(np.abs(X - target_value))


def get_n_largest_values(X, n):
    return X[X.argsort()[::-1][:n]]