import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    dataset = open(filename, "r")
    rating_csv = pd.read_csv(dataset)

    rating_unstacked = rating_csv.groupby(['source', 'target'])['rating'].first().unstack().fillna(0)
    answer = rating_unstacked.to_numpy(dtype='float32')
    # answer = rating_un.values
    return answer

def get_frequent_matrix(filename, dtype=np.float32):
    dataset = open(filename, "r")
    frequent_csv = pd.read_csv(dataset)
    
    frequent_csv['count'] = 1
    frequent_csv = frequent_csv.groupby(['source', 'target']).sum().unstack().fillna(0)
    answer = frequent_csv.to_numpy(dtype='float32')
    # answer = rating_un.values
    return answer