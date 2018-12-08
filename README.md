"# Stock-Price-Predictor" 

import pandas as pd
import quandl, math, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

'''
Stock price predictor based on LinearRegression and achieves accuracy of up to 99.05% on Coca Cola.co
Dataset is obtained from quandl
'''

df = quandl.get("EOD/KO", authtoken="h2rGzLgJ8dnFfsfXtDAV")	
#get dataframe from quandl ticker

df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]
#restrict collumns to the following

df['HL_pct'] = ((df['Adj_High'] - df['Adj_Low']) / df['Adj_Low'])*100
df['Pct_change'] = ((df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'])*100
#Create 2 new data collumns HL_pct and Pct_change

new_df = df[['Adj_Close','HL_pct', 'Pct_change', 'Adj_Volume']]
#set new dataframe with relevant features

forecast_col = 'Adj_Close'

new_df.fillna('-99999', inplace=True)
#treat rows with missing data as outlier

forecast_out = int(math.ceil(0.005*len(df)))
#set forecast length to be 0.5% of dataframe length

new_df['label'] = df[forecast_col].shift(-forecast_out)
#creates new collumn with adj_close data shifted upwards by forecast length
#this means all features is attached to a label, adj_close x number of days into the future


X = np.array(new_df.drop(['label'], 1))
#first array -> features only
#removes 'label' collumn from dataframe and assigns arrayX with new dataframe returned
#the '1' in the drop method specifies the axis of the dataframe that will be dropped, i.e. the top-down axis
X = preprocessing.scale(X)
#scales data in X (to what scale specifically?)
X_withoutlabels = X[-forecast_out:]
X_withlabels = X[:-forecast_out]

y = np.array(new_df['label'])
#second array -> label
y = y[:-forecast_out]

''' #Training process only needs to be done the first time. Use pickle to store trained linear regression
X_train, X_test, y_train, y_test = train_test_split(X_withlabels, y, test_size = 0.1)
#shuffles and splits data into training and testing with test size = 20% of total data

clf = LinearRegression(n_jobs=-1)
#use linearRegression as classification model
#always check documentation to see if model can accept parameter 'n_jobs'
#n_jobs means number of threads run by processor at any point in time, -1 is to use as many as possible by your cpu
clf.fit(X_train, y_train)
#use X-train and y_train to train model
with open('CocaColaLinearRegression','wb') as f:
	pickle.dump(clf, f) #saves classifier object into a pickle to cut down on training time

accuracy = clf.score(X_test, y_test)
#tests model using X_test and y_test then get accuracy
'''

file = open('CocaColaLinearRegression','rb')
clf = pickle.load(file)

forecast_set = clf.predict(X_withoutlabels)
#print(accuracy, forecast_out)
new_df['Forecast'] = np.nan

last_date = new_df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	new_df.loc[next_date] = [np.nan for a in range(len(new_df.columns)-1)] + [i]
	#new_df.loc creates a new row if the index does not exist
	#assign  nan collumns to each row of the new dates except label column where i value in forecast_set is assigned
style.use('ggplot')
new_df['Adj_Close'].plot()
new_df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
