
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import pearsonr
import math,sys,os
import random as rnd
from scipy.misc import derivative
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, regularizers
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Functions import create_dataset, MakeNoise
#####################################################################
### run as python -W ignore main.py DataName.txt
### For example python -W ignore main.py 'deterministic_chaos.txt' 0 1
# load the dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NomeFile = sys.argv[1]
dataset = np.matrix(read_csv(NomeFile, sep=" ", header=None))
########################################################################################
train_length = 100
validation_length = 50
test_length = 30

tstart =  int(sys.argv[3])

### Take the training set
ts_training = dataset[tstart:(tstart + train_length + validation_length),:]
ts_training = preprocessing.scale(ts_training)
Noise = int(sys.argv[2])
if Noise == 1:
        ts_training = MakeNoise(ts_training, 0.2)
        print('The time series has been contaminated with observational noise')
        print('However, you check if you correctly predict the noise-free time series in the test set')
###
num_species = ts_training.shape[1]
#### Give a different representation of the training set
ts_training_original = ts_training
#ts_training = StackDAE(ts_training, train_length, validation_length, 5, dim_red = 0)
#### Reshape into X=t and Y=t+look_back
look_back = 1
### Here you create an array Ytrain with the column to predict scale by look_back points (e.g.,, 1)
ts_training_tr = ts_training[0:train_length,:]
tr_training_vl = ts_training[train_length:(train_length + validation_length),:]
trainX, trainY = create_dataset(ts_training_tr, look_back)
ValX, ValY = create_dataset(tr_training_vl, look_back)
####################################################################################
test_set = dataset[(tstart + train_length + validation_length):(tstart + train_length + validation_length + test_length), :]
test_set = preprocessing.scale(test_set)
####################################################################################
#### Take last point of the training set and start predictions from there
last_point_kept = ts_training[(np.shape(ts_training)[0] - 1), :]
#####################################################################################
###### Initialise the autoencoder
#### Some properties of the autoencoder
encoding_dim = np.shape(trainX)[1]
## This is the size of the decoder (dimension of the state space)
decoding_dim = np.shape(trainX)[1]
###########################################################################
input_ts = Input(shape = (decoding_dim,))
###########################################
#### Decide whether to use saprsity or not
#encoded = Dense(encoding_dim, activation= 'sigmoid', activity_regularizer=regularizers.l2(10e-3))(input_ts)
###########################################
encoded = Dense(encoding_dim, activation= 'sigmoid', activity_regularizer=regularizers.l2(10e-5))(input_ts)
decoded = Dense(decoding_dim, activation= 'linear', activity_regularizer=regularizers.l2(10e-5))(encoded)
#decoded = Dense(decoding_dim, activation= 'linear', activity_regularizer=regularizers.l2(10e-3))(encoded)

autoencoder = Model(input_ts, decoded)
encoder = Model(input_ts, encoded)
# create a placeholder for an encoded (d-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# choose your loss function and otpimizer
autoencoder.compile(loss='mean_squared_error', optimizer='adam')
########################
#### Train the autoencoder but avoid writing on stdoutput
autoencoder.fit(trainX, trainY,
                        epochs= 400,
                        batch_size = 6,
                        shuffle = False,
                        validation_data=(ValX, ValY), verbose = 0)
# make predictions
length_predictions = test_length
realizations = 20
next_point = np.zeros((length_predictions,num_species))
for prd in range(realizations):
	##### Last point of the training set for predictions
	last_point  = last_point_kept.reshape((1,num_species))
	##
	encoded_ts = encoder.predict(last_point)
	last_point = decoder.predict(encoded_ts)
	next_point[0,:] = next_point[0,:] + last_point
	##
	for i in range(1,length_predictions):
		encoded_ts = encoder.predict(last_point)
		last_point = decoder.predict(encoded_ts)
		next_point[i,:] = next_point[i,:] + last_point

next_point = next_point/realizations
next_point = np.delete(next_point, (0), 0)
########### Training data
encoded_ts = encoder.predict(ts_training)
training_data = decoder.predict(encoded_ts)
training_data = np.insert(training_data, 0, np.array(np.repeat('nan',num_species)), 0) 


os_rmse = np.sqrt(np.mean((next_point - test_set[1:(length_predictions),:])**2))
os_correlation = np.mean([pearsonr(next_point[:,i], test_set[1:(length_predictions), i])[0] for i in range(num_species)])


print 'RMSE of LSTM  forecast = ', os_rmse
print 'correlation coefficient of LSTM  forecast = ', os_correlation

########################################################################################################
plot = True
if plot == True:
        all_data = np.concatenate((ts_training_original,test_set[0:(length_predictions),:]), axis = 0)
        all_data_reconstructed = np.concatenate((training_data,next_point), axis = 0)
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	interval_forecast = range((train_length + validation_length+1), np.shape(all_data_reconstructed)[0])

	ax1.plot(all_data[:,0], color = 'b')
	ax1.plot(interval_forecast, all_data_reconstructed[interval_forecast,0], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax1.axvline(x = (train_length + validation_length), lw = 2, ls = '--')
	ax1.legend()

	ax2.plot(all_data[:,1], color = 'b')
	ax2.plot(interval_forecast, all_data_reconstructed[interval_forecast,1], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax2.axvline(x = (train_length + validation_length), lw = 2, ls = '--')

	ax3.plot(all_data[:,2], color = 'b')
	ax3.plot(interval_forecast, all_data_reconstructed[interval_forecast,2], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax3.axvline(x = (train_length + validation_length), lw = 2, ls = '--')

	ax4.plot(all_data[:,3], color = 'b')
	ax4.plot(interval_forecast, all_data_reconstructed[interval_forecast,3], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax4.axvline(x = (train_length + validation_length), lw = 2, ls = '--')
	plt.show()
