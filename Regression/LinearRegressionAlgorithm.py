from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)


#CREATING LINEAR DATASET FOR TESTING 
#hm --> how many data points do you want 
#variance --> how variable do you want this dataset to be
#step --> how far on average to step up the y value 
#correslation --> do you want the data to be correlated positively or negatively or just not correlated at all 
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        #this will give us data but with no correlation between the different data points 
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        #taking correlation into account 
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
    m = ( (mean(xs)*mean(ys)) - mean(xs*ys) ) / ( (mean(xs)**2) - (mean(xs**2)) )
    b = mean(ys) - m*mean(xs)
    return m, b 

xs, ys = create_dataset(40, 10, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs,ys)
#print(m, b)

regression_line = [(m*x)+b for x in xs]

#R Squared Theory (r^2 metric) for accuracy 
#We will calculate accuracy for the algorithm with Squared Error 
#The error is the distance between the point (actual value) and the best fit line
#We square the error so that we only have to deal with positive values
#We also square because we want to penalize for outliers (points very far away from best fit line) 
#We want our r^2 value to be as high as possible 

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1-(squared_error_regr / squared_error_mean)


predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()






