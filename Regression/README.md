<!--
*** Markdown
*** coding: utf-8
*** Author: Malek Kchaou
*** Date: 05-28-2023 
*** Last Modified time: 05-29-2023
*** Last Modified by: Malek Kchaou 
-->

## Linear Regression: 

In this folder, there are two main source code files:
  * StockPrediction.py
  * LinearRegressionAlgorithm.py

---

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
**We dropped data samples that will be used in the predictions which represented the last 343 days of the original dataset. What we were left with was then used for training and testing.  
So the features used for training and testing were dating from 08-19-2004 to 11-10-2016 and the labels from 11-11-2016 to 03-27-2018**

After adjusting the dataset and adding the 'label' column, this what the newly generated dataset looked like:

![google_dataset2](https://github.com/MK720-dev/Machine-Learning-with-Python-Concepts-and-Applications/assets/78389944/6d372e21-184f-4955-98ca-2c890d4b9d47)

**Important:** --> It is also important to note that missing data was dealt with by setting eveyr NaN value to -99999 which makes it automatically seen as an outlier by the linear regression model
               --> Features were also rescaled and normalized to have values between -1 and 1
               
### Training and Testing 

A scikit-learn linear regression classifer was used to train and test the prediction model. 
<ins>**The model was trained to use the features from 343 days ago to predict the Adjusted Close price in 343 days.**</ins>

We used numpy's **model_selection.train_test_split()** function to divide the dataset into training and testing sets. The testing set's size was set to be 20% of the whole dataset. 

**clf.score()** was used to get the model's **accuracy**. 
--> over the span of 25 simulations, the average accuracy was **88%**.

### Predictiions:

Predictions were made based on features dating from **11-11-2016 to 10-18-2017** (11-10-2016 + 343 days) and predictions were made for the dates ranging between **10-20-2017 to 09-26-2018**

Here's the final plot of the Adjusted Close prices (feature), label and the forecast results:

![final_plot](https://github.com/MK720-dev/Machine-Learning-with-Python-Concepts-and-Applications/assets/78389944/961acb78-916d-4f98-8c3b-49db5461a47b)

**Important:** note that label and forecast graphs are squished back by 343 days

### <ins>LinearRegressionAlgorithm.py</ins>








