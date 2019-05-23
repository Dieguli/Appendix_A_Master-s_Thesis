import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.tools import freeze_graph
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import os
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import L1L2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

#------START PREPROCESSING--------------------
#All the prerpocessing codde has been take from : https://github.com/Cerebro409/EEG-Classification-Using-Recurrent-Neural-Network
import random
seed=42
random.seed(seed)
eeg1 = pd.read_csv("eeg1.csv", delimiter="\t")
new_columns = eeg1.columns.values 
new_columns[0] = 'time'     
new_columns[33] = 'sample' 
eeg1.columns = new_columns
events1 = pd.read_csv("events1.csv")
eeg1_smol = eeg1[0:785000]
events1_smol = events1[0:1000]
def generate_eeg(samples, time_steps, n_features, event_types):
    signals = generate_signals(samples, time_steps, n_features)
    events = generate_events(event_types, samples)
    events_1hot = one_hot_events(events)
    return signals, events_1hot
def generate_signals(samples, time_steps, n_features):
    signals = np.random.random((samples, time_steps, n_features))
    return signals
def generate_events(event_types, samples):
    events = np.random.randint(1, event_types, samples)
    return events
def clean_eeg(eeg, events, event_interval_length, eeg_slice_length):
    array_list = [] 
    index_list = []
    eeg = standardize_eeg(eeg) 
    for index, row in itertools.islice(events.iterrows(), event_interval_length): 
        tmin, tmax = build_event_intervals(row, events)
        eeg_slice = cut_event_intervals(eeg, tmin, tmax)
        array_list, index_list = build_array(eeg_slice, eeg_slice_length, 
                                             index, index_list, array_list)
    y_int = events.iloc[index_list] 
    y_int = y_int['type'].values    
    X = np.stack(array_list, axis = 0)   
    return X, y_int                     
    
        
def build_event_list(row, event_list):
    event_type = getattr(row, "type")
    event_list.append(event_type)
        
def build_event_intervals(row, events):
    tmin = getattr(row, "latency")
    tmin_in = getattr(row, "number")
    tmax_in = tmin_in + 1
    tmax = events1.loc[tmax_in, "latency"]
    return tmin, tmax

def cut_event_intervals(eeg, tmin, tmax):
    eeg_slice = eeg.loc[(eeg["time"] > tmin) & (eeg["time"] < tmax)]
    eeg_slice.drop(["time", "sample"], axis = 1, inplace = True)
    return eeg_slice
    
def build_array(eeg_slice, eeg_slice_length, index, index_list, array_list):
    if len(eeg_slice) < eeg_slice_length:
        index_list.append(index)
        eeg_matrix = eeg_slice.as_matrix()
        padded_matrix = np.pad(eeg_matrix, ((0, eeg_slice_length - len(eeg_matrix)), (0,0)),
                                   'constant', constant_values=0)
        array_list.append(padded_matrix)
    return array_list, index_list

def one_hot_events(events):
    events_list = list(events)
    lb = preprocessing.LabelBinarizer()
    lb.fit(events_list)
    events_1hot = lb.transform(events_list)
    return events_1hot, lb

def invert_one_hot(events, lb):
    inv_events = lb.inverse_transform(events)
    return inv_events
def standardize_eeg(eeg_data):
    column_list = eeg_data.columns[1:33]
    time = eeg_data['time']
    sample = eeg_data['sample']
    eeg_array = eeg_data[column_list]
    eeg_stnd = scale_data(eeg_array)
    eeg_stnd_df = pd.DataFrame(eeg_stnd, index=eeg_data.index, columns=column_list)
    eeg_stnd = pd.concat([time, eeg_stnd_df, sample], axis =1)
    return eeg_stnd
def scale_data(unscaled_data):
    scaler = StandardScaler()
    scaler.fit(unscaled_data)
    scaled_data = scaler.transform(unscaled_data)
    return scaled_data
import math
time_steps = 1300
def build_zero_events(event_data, time_steps=time_steps):
    new_events = build_new_events(event_data, time_steps)
    events = zero_events(event_data, new_events)
    return events
def build_new_events(event_data, time_steps= time_steps):
    first_event_time = event_data['latency'].loc[1]
    number_new_intervals = math.floor(first_event_time / time_steps)
    df = pd.DataFrame(columns=['number', 'latency', 'type', 'duration'],index = range(number_new_intervals) )
    latency = 0
    for t in range(number_new_intervals):
        latency += 1300
        df.loc[t].latency = latency
        df.loc[t].type = 0
    return df
def zero_events(event_data, new_events):
    events_zeros = event_data[event_data.latency != 1]
    events_zeros= new_events.append(events_zeros)
    events_zeros = events_zeros.reset_index(drop=True)
    events_zeros['number'] = events_zeros.index + 1
    return events_zeros

samples = 3625  
n_features = 32  
time_steps = 1300 
event_types = 2 
X, y = clean_eeg(eeg1, events1, samples, time_steps)  
remove_list = [0,2,4,5,6]              
drop_list = np.isin(y, remove_list)                    
drop_array = np.array(drop_list)       

y_short_int = y[np.isin(y,remove_list, invert=True)]
X_short = X[np.isin(y, remove_list, invert=True)]
y_short, lb = one_hot_events(y_short_int)
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
sss.get_n_splits(X_short, y_short)



for train_index, test_index in sss.split(X_short, y_short):
    X_train, X_test = X_short[train_index], X_short[test_index]
    y_train, y_test = y_short[train_index], y_short[test_index]

print(X_train.shape)

print(X_test.shape)
print(y_train.shape)

print(y_test.shape)

#----------FINISH PREPROCESSING--------------------

#----------MODEL-----------------------------------
tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
model = Sequential()
model.add(LSTM(100, return_sequences=False, input_shape=(time_steps, n_features),kernel_regularizer=L1L2(0.0, 0.00001)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(0.0, 0.00001)))
optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=16, epochs=50,  callbacks = [tensorboard]) 
score = model.evaluate(X_test, y_test, batch_size=16) 
#---------------------------------------------------
#---------CONVERT MODEL TO .pb FORMAT---------------
