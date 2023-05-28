import numpy as np
from math import sqrt
import warnings 
import matplotlib.pyplot as plt 
from matplotlib import style 
#this will be used for votes during prediction  
from collections import Counter

style.use('fivethirtyeight')

#simple dataset to begin with 
dataset = {'k' : [[1,2], [2,3], [3,1]], 'r' : [[6,5], [7,7], [8,6]]}
new_feature = [5,7]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            #problem with this version of the algorithm is that it doesn't run fast and it's hard coded to only do calculations in 2D 
            #euclidean_distance = sqrt((predict[0]-features[0])**2 + (predict[1]-features[1])**2)
            #we could use numpy arrays and sum function
            #euclidean_distance = np.sqrt(np.sum((np.array(predict)-np.array(predict))**2))
            #or we could just use an even simpler code:
            euclidean_distance = np.linalg.norm(np.array(predict)-np.array(features))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result 

result = k_nearest_neighbors(dataset, new_feature)
print(result)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100, color=result)
plt.show()