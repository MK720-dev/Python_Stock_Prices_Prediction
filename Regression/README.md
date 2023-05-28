## Linear Regression: 

In this folder, there are two main source code files:
  * StockPrediction.py
  * LinearRegressionAlgorithm.py

### <ins>StockPrediction.py</ins>

In this program, I used a scikit-learn Linear Regression model to predict Google Stock prices. 

#### Dataset:
I used python's **Quandl** module to import a real life **Google Stock prices** dataset from the core financial datasets on the **Nasdaq Data Link** website. 
The dataset contains Google stock prices from **08-19-2004** up to **03-27-2018** 

For simplicity reasons I will only include the first five lines of the dataset in the accompanying images. 

This is how the dataset initially looked like: 

![Google_dataset1](https://github.com/MK720-dev/Machine-Learning-with-Python-Concepts-and-Applications/assets/78389944/9392e684-5444-4e05-aeac-01ca7e33dfbb)

I then proceeded to define two new features:
 - The High Low Percentage (HL_PCT)
 - The Percentage of Change (PCT_change)
```
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100
```
I decided to predict 1% of the entire dataframe and for that I calculated the corresponding amount of days: 
```
forecast_out = int(math.ceil(0.1*len(df)))
```
forecast_out evaluated to 343 days --> **algorithm will predict Stock prices for almost a year ahead (11,2767 months)**

I chose the Adjusted Close price **'Adj. Close'**, the High Low Percentage **'HL_PCT'**, the Percentage of Change **'PCT_change'** and the **'Adjusted Volume'** to be the features of my linear regression model. 
As for the the labels that were used to train the model, I went for the Adjusted Close price 'Adj. Close' in 343 days. 
<ins>**Thus, the model was trained to use the features from 343 days ago to predict the Adjusted Close price in 343 days.**</ins>

After adjusting the dataset and adding the labels columns, this what the newly generated dataset looked like: 

![google_dataset2](https://github.com/MK720-dev/Machine-Learning-with-Python-Concepts-and-Applications/assets/78389944/6d372e21-184f-4955-98ca-2c890d4b9d47)









