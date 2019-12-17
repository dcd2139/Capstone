#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import random
import matplotlib.pyplot as plt
import cftime
import xarray as xr 
import numpy as np
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop

LOGDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/baseline/logs/"
MODELDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/baseline/model/model.h5"
CHECKPOINTDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/baseline/model/checkpoints/"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOGDIR)

DS = xr.open_dataset("30_Min_Interval/000_train.nc", decode_times=False, decode_cf=False)

BATCH_SIZE = 100
NUM_EPOCHS = 5

nparray = DS.variables["vars"].values
print(nparray.shape)

# Define model

MODEL_INPUT_FEATURES = 64
TRAINING_OPTIMIZER = 'rmsprop'
TRAINING_LOSS = 'mse'

model = Sequential()
model.add(Dense(128, input_dim=MODEL_INPUT_FEATURES))
model.add(LeakyReLU(alpha=0.25))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.25))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.25))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.25))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.25))
model.add(Dense(1, activation='relu'))
opt = RMSprop(learning_rate=0.0001)
model.compile(optimizer=opt, loss=TRAINING_LOSS)

# Model checkpointing

filepath="weights.{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(CHECKPOINTDIR + filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


# Select all data for given lat lon

in_series = nparray[::,0:64:1]

out_series = nparray[::,128:129:1]

# fit model
model.fit(x=in_series, y=out_series, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, callbacks=[tensorboard_callback, checkpoint])

model.save(MODELDIR)
del model 
