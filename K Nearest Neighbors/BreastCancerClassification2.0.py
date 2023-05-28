import numpy as np
from math import sqrt
import warnings   
from collections import Counter
import pandas as pd
import random

df = pd.read_csv(r'C:\Users\kchao\OneDrive\Documents\Dossier Malek\ML with Python\K Nearest Neighbors\breast-cancer-wisconsin.data')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(predict)-np.array(features))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result 


