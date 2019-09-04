import datetime

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import numpy as np

df = pd.read_csv("data/AAPL.csv")
print(df.tail(), len(df))

# Remove any rows with null data
df = df.dropna()

print(len(df))

# Smooth the data to cute out some noise
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
plt.show()


# Drop missing value
df.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(df)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
df['label'] = df[forecast_col].shift(-forecast_out)
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']
df = df.drop(['Date'], 1)
print(df.head())
# y = df['Date']
# df = df.drop(['Date'], 1)
X = np.array(df.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(df['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape)

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

# SGD Regression
clfsgd = SGDRegressor()
clfsgd.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)
confidencesgd = clfsgd.score(X_test, y_test)
# results
print('The linear regression confidence is ', confidencereg)
print('The quadratic regression 2 confidence is ', confidencepoly2)
print('The knn regression confidence is ', confidenceknn)
print('The sgdr regression confidence is ', confidencesgd)

# The linear regression confidence is  0.9658729472274955
# The quadratic regression 2 confidence is  0.9672571704432426
# The knn regression confidence is  0.9635175964516575
# The sgdr regression confidence is  0.9611887643174606

from sklearn.model_selection import GridSearchCV

params = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['none','l2','l1','elasticnet'],
    'alpha': [.0001, .001, .01, .00001],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
}

clf = GridSearchCV(SGDRegressor(), param_grid=params, cv=5, scoring='r2', n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_params_)
confidencegrid = clf.best_estimator_.score(X_test, y_test)

print('The gridsearch confidence is ', confidencegrid)

# Best params
# {'alpha': 1e-05, 'learning_rate': 'adaptive', 'loss': 'squared_epsilon_insensitive', 'penalty': 'elasticnet'}
# The gridsearch confidence is  0.9644860699859358

forecast_set = clf.best_estimator_.predict(X_lately)
df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Adj Close'].tail(500).plot()
df['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Viewable at ./prediction_results.png