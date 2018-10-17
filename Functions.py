from keras.layers import Input, Dense, regularizers
from keras.models import Model
import numpy as np
from scipy import stats
from sklearn import preprocessing
import sys, os
import math,sys,os
import random as rnd
from keras.models import Sequential
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


### Function to create the lagged dataset
def create_dataset(dtset, look_back=1):
	### Input dtset = time series
	### look_back = time lage of predictions
	### Output: two data set one the same and one lagged
        dataX = np.zeros((np.shape(dtset)[0] - look_back - 1, np.shape(dtset)[1]))
        dataY0 = []; dataY1 = []; dataY2 = []; dataY3 = [];
        dataY = np.zeros((np.shape(dtset)[0] - look_back - 1, np.shape(dtset)[1]))
        for i in range(np.shape(dtset)[0] - look_back - 1):
                dataX[i,:] = dtset[i:(i+look_back), :]
                dataY[i,:] = dtset[i+look_back,:]
        return np.array(dataX), np.array(dataY)

def MakeNoise(X, percentage_std):
        ### Input: X = training set of the time series
        ###        percentage_std = percentage of standard deviation for the noise [0,1]
        X = X + np.random.normal(loc=0.0, scale=percentage_std*np.std(X[:,1]), size= X.shape)
	return(X)

def parameters(enc, dec):
	### Input: enc,dec = encoder,decoder
	### Output:
	for layer in enc.layers:
		weights_encoder = layer.get_weights()
	for layer in dec.layers:
		weights_decoder = layer.get_weights()
	Wh = weights_encoder[0]
	bh = weights_encoder[1]
	Wg = weights_decoder[0]
	bg = weights_decoder[1]

	return(Wh, bh, Wg, bg)

