#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import pandas as pd
import random
import matplotlib.pyplot as plt
import cftime
import xarray as xr 
import numpy as np
import keras
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

LOGDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/v7/logs/"
MODELDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/v7/model/model.h5"

DS = xr.open_dataset("30_Min_Interval/000_valid.nc", decode_times=False, decode_cf=False)

NUM_LAT_LON = 13824
TIMES = 431
NUM_SAMPLES = TIMES * NUM_LAT_LON
INPUT_TIME_STEPS = 10
INPUT_FEATURES = 64

nparray = DS.variables["vars"].values
print(nparray.shape)

# Load model

model = load_model(MODELDIR)

all_ys = []
all_yhats = []

# loop over each lat lon
for start_index in range(0,NUM_LAT_LON):

    indices = []
    
    for i in range(start_index,NUM_SAMPLES,NUM_LAT_LON):
    
        indices.append(i)
    
    print(f"start_index {start_index} found {len(indices)} indices")

    selectedvalues = nparray[tuple(indices),]

    x = []
    y = []
    yhat = []
    
    for time in range(0, TIMES-INPUT_TIME_STEPS-1):

        valid_series = selectedvalues[time:time+INPUT_TIME_STEPS:1,0:INPUT_FEATURES:1]

        #print(f"Valid output at {time+INPUT_TIME_STEPS} and {time+INPUT_TIME_STEPS+1}")
        valid_output = selectedvalues[time+INPUT_TIME_STEPS:time+INPUT_TIME_STEPS+1:1,128:129:1]

        x.append(time)
        y.append(np.asscalar(valid_output))
        all_ys.append(np.asscalar(valid_output))

        prediction = model.predict(valid_series.reshape((1, INPUT_TIME_STEPS, INPUT_FEATURES)), verbose=0)
        yhat.append(np.asscalar(prediction))
        all_yhats.append(np.asscalar(prediction))
   

    if (start_index % 1000 == 0):	     
        PLOTFILE = f"/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/v7/validate/{start_index}.png"
        
        plt.plot(x,y)
        plt.plot(x,yhat)
        plt.legend(['y = actual', 'y = prediction'], loc='upper left')
        
        plt.savefig(PLOTFILE)
        plt.close()
    data = [yhat,y]
    print(np.corrcoef(y,yhat))

zipped_list =  list(zip(all_ys, all_yhats))
df=pd.DataFrame(data= zipped_list, index=[i for i in range(len(all_ys))], columns=['y','yhat'])
 
print(df.describe())

print(f"MSE = {mean_squared_error(all_ys, all_yhats)}")

