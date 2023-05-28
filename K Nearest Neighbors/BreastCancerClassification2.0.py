import numpy as np
from math import sqrt
import warnings   
from collections import Counter
import pandas as pd
import random

accuracies = []

for i in range(25):
    df = pd.read_csv(r'C:\Users\kchao\OneDrive\Documents\Dossier Malek\ML with Python\K Nearest Neighbors\breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    #some values in the dataframe are coming in with quotes
    #we have to make sure all values are floats, then we convert the dataframe to a list of list 
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data) 

    #create our training and testing sets 
    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]


    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    def k_nearest_neighbors(data, predict, k=3):
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups!')
        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(predict)-np.array(features))
                distances.append([euclidean_distance, group])

        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1]/k
        #print(vote_result, confidence)
        return vote_result, confidence 

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            #Adding to k doesn't necessarily increase accuracy 
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            else:
                print(confidence)
            total += 1

    accuracy = correct/total
    print('Accuracy: ', accuracy )
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))

#if we compare to the first version (1.0) of the program where I used scikit-learn,
#we can see that we can pretty similar and comparable accuracies if we repeat the training 
#and testing process the same amount of times
#however the sklearn version is faster since it can be threaded using n_jobs parameter 
#you can use a radius to make it run faster 
#general note 1: (about K Nearest Neighbors) it scales really well to decent amounts of datasets, 
#but it is still not well suited to run on extremely big datatsets 
#general note 2: we could use linear regression fro classification as long as the dataset is linear 



