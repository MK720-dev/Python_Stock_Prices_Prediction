import pandas as pd 
import quandl, math, datetime
import numpy as np 
import pickle #Pickling will allow us to save the classifier and use without having to re-train it everytime 

#We need preprocessing becasue we watn our feature values to be between [-1,1] 
#This helps with the accuracy and computing speed
#train_test_split is used to create training and testing samples.
#it also helps separate data
#svm because you can use support vector machines for regression --> I ended up not using svms
from sklearn import preprocessing, svm  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style 

style.use('ggplot')

accuracies = []

for i in range(25):
    quandl.ApiConfig.api_key = 'zZEPoAwdXYRymNs6tVUe'

    df = quandl.get('WIKI/GOOGL')

    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
    df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100

    #Better feature than these could be chosen for model training 
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

    #regression forecast column
    forecast_col = 'Adj. Close'

    #fill in missing data
    #setting missing data to -99999 makes them outliers
    #they will be automatically neglected by the algorithm
    df.fillna(-99999, inplace=True)

    #predict label value after a specific number of days 
    #number of days = forecast_out (1% of length of dataframe) 
    forecast_out = int(math.ceil(0.1*len(df)))

    #each row's label value will be the Adjusted Close a certain amount of days (our forecast_out) into the future 
    df['label'] = df[forecast_col].shift(-forecast_out)
   
    X = np.array(df.drop(['label'], axis= 1))
    #Scaling before feeding to classifier 
    #Normalized with all the other data points 
    #In order to scale properly you will need to scale new values alongside the training set 
    X= preprocessing.scale(X)
    X_lately = X[-forecast_out:] 
    X = X[:-forecast_out]

    #remove rows with at least one NaN value
    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #we can set n_jobs to a non default value to make use of multithreading 
    #-1 will use as many jobs as the computer processor is able to run 
    clf = LinearRegression(n_jobs=-1)
    #train the linear model
    clf.fit(X_train, y_train)
    #Pickling: Save classifier to binary file 
    pickle_file_path= r'C:\Users\kchao\OneDrive\Documents\Dossier Malek\ML with Python\Regression\linearregression.pickle'
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(clf, f)

    #Use the saved classifier 
    pickle_in = open(pickle_file_path, 'rb')
    clf = pickle.load(pickle_in)

    #test the linear model
    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)

    #predict 
    forecast_set = clf.predict(X_lately)

    #print(forecast_set, accuracy, forecast_out)

    df['Forecast'] = np.nan

    #df.iloc is index based 
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp() #timestamp is pandas equivalent of python's Datetime
    one_day =  86400
    next_unix = last_unix + one_day 

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        #df.loc is label based 
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

    #print(df.to_string())


print(sum(accuracies)/len(accuracies))

#Plotting 
df['Adj. Close'].plot()
df['label'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
