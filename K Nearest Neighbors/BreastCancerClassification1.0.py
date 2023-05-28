#K Nearest Neighbors is bad with huge datasets and handles outliers very poorly
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd 

df = pd.read_csv(r'C:\Users\kchao\OneDrive\Documents\Dossier Malek\ML with Python\K Nearest Neighbors\breast-cancer-wisconsin.data')
#replace missing data 
df.replace('?', -99999, inplace=True)
#drop irrelevant parameter 
#including the id column will make the accuracy drop drastically
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'],axis=1))
y = np.array(df['class'])
#create training and testing sets 
X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.2)

#Create the K Nearest Neighbors classifier, train and test it
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
#reshape numpy array to (1,9) --> one sample or (2,9) --> two samples or however many samples you want to make a specific number of predictions 
example_measures = example_measures.reshape(len(example_measures),-1)
#predict 
prediction = clf.predict(example_measures)
print(prediction)

