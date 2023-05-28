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
 - The High Low Percentage
 - The Percentage of Change
```
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100
```










