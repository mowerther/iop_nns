"""
Script for loading data and training a Recurrent Neural Network (RNN).
Trains N networks, evaluates them, and saves their outputs.
"""
import pandas as pd
import pnn

### LOAD DATA
# Load from file
train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood = pnn.read_all_data()
data_train, data_test = train_set_random, test_set_random
print("Loaded data")

# Select Rrs values in 5 nm steps, IOP columns
X_train, y_train = pnn.nn.extract_inputs_outputs(data_train)
X_test, y_test = pnn.nn.extract_inputs_outputs(data_test)

# Rescale y data (log, minmax)
y_train_scaled, y_test_scaled, scaler_y = pnn.nn.scale_y(y_train, y_test)
print("Rescaled data")

# RNN: reshape data
X_train_reshaped, X_test_reshaped = pnn.nn.rnn.reshape_data(X_train, X_test)


### RNN TRAINING
model = pnn.nn.rnn.build_and_train_rnn_mcd(X_train_reshaped, y_train_scaled)
print("Trained model")
