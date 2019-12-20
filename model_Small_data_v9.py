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
from keras.layers import LSTM
from keras.layers import Activation
from keras.optimizers import RMSprop 
from keras.callbacks import ModelCheckpoint

LOGDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/v9/logs/"
MODELDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/v9/model/model.h5"
CHECKPOINTDIR = "/Users/dhananjaydeshpande/Desktop/Columbia Data Science/Capstone/v9/model/checkpoints/"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOGDIR)

DS = xr.open_dataset("Small_data/000_train.nc", decode_times=False, decode_cf=False)

NUM_LAT_LON = 13824
TIMES = 855 * NUM_LAT_LON 
INPUT_TIME_STEPS = 10
BATCH_SIZE = 50
STEPS_PER_EPOCH = 10
NUM_EPOCHS = 10

nparray = DS.variables["vars"].values
print(nparray.shape)

# Define model

MODEL_LSTM_UNITS = 10
MODEL_LSTM_ACTIVATION = 'relu'
MODEL_INPUT_FEATURES = 64
MODEL_DENSE_OUTPUTS = 1
MODEL_DENSE_ACTIVATION = 'relu'
TRAINING_OPTIMIZER = 'RMSprop'
TRAINING_LOSS = 'mse'

model = Sequential()
model.add(LSTM(MODEL_LSTM_UNITS, activation=MODEL_LSTM_ACTIVATION, input_shape=(INPUT_TIME_STEPS, MODEL_INPUT_FEATURES)))
model.add(Dense(MODEL_DENSE_OUTPUTS))
model.add(Activation(MODEL_DENSE_ACTIVATION))
rms = RMSprop(learning_rate=0.0001, rho=0.9)
model.compile(optimizer=rms, loss=TRAINING_LOSS)

# Model checkpointing

filepath="weights.{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(CHECKPOINTDIR + filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

# loop over each LAT LON
for start_index in range(0,100):

    indices = []
   
    # Get all indices for the LAT LON 
    for i in range(start_index,TIMES,NUM_LAT_LON):
    
        indices.append(i)
    
    print(f"start_index {start_index} found {len(indices)} indices")

    # select in chunks of 96 for each day of LAT LON for upto 9 days
    for range_index in range(0, 9):
        
        selected_indices = [i for i in range(range_index*96, 95+(range_index*96))]
           
        final_selections = []    
        for j in range(0,96):
            if len(indices) > 0:
                final_selections.append(indices.pop(0))
    
        #final_selections = [indices[i] for i in selected_indices]
        
        print(f"final selections are {final_selections}")
        
        # Remove index out of bound
        if 11819520 in final_selections:
            final_selections.remove(11819520)

        print(f"Current batch found {len(final_selections)} indices")

        selectedvalues = nparray[tuple(final_selections),]
        # Select all data for given lat lon
    
        in_series = selectedvalues[::,0:64:1]

        out_series = selectedvalues[::,128:129:1]

        # Shift out series values
        out_series = np.insert(out_series, 0, 0)
        out_series = out_series[0:-1]

        # define generator
        generator = TimeseriesGenerator(in_series, out_series, length=INPUT_TIME_STEPS, batch_size=BATCH_SIZE)

        n_features = in_series.shape[1]

        # fit model
        if start_index % 1000 == 0:
            model.fit_generator(generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUM_EPOCHS, verbose=1, callbacks=[tensorboard_callback, checkpoint])
        else:
            model.fit_generator(generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUM_EPOCHS, verbose=0, callbacks=[tensorboard_callback, checkpoint])

model.save(MODELDIR)
del model 
