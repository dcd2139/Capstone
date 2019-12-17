#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import random
import matplotlib.pyplot as plt
import cftime
import xarray as xr 
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

LOGDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/baseline/logs/"
MODELDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/baseline/model/model.h5"

DS = xr.open_dataset("30_Min_Interval/000_valid.nc", decode_times=False, decode_cf=False)

nparray = DS.variables["vars"].values
print(nparray.shape)

# Load model

model = load_model(MODELDIR)

# loop over each lat lon

x = nparray[0:1000000:1,0:64:1]
y = nparray[0:1000000:1,128:129:1]

times = [count for count in range(0,1000000)] 

predictions = model.predict(x=x, verbose=1)
yhat = list(predictions)
        
PLOTFILE = f"/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/baseline/validate/result.png"

plt.plot(times,y)
plt.plot(times,yhat)
plt.legend(['y = actual', 'y = prediction'], loc='upper left')
        
plt.savefig(PLOTFILE)
