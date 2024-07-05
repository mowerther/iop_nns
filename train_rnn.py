"""
Script for loading data and training a Recurrent Neural Network (RNN).
Trains N networks, evaluates them, and saves their outputs.
"""
import pnn

### LOAD DATA
# Load from file
train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood = pnn.read_all_data()
data_train, data_test = train_set_random, test_set_random
print("Loaded data.")

# Select Rrs values in 5 nm steps, IOP columns
X_train, y_train = pnn.nn.extract_inputs_outputs(data_train)
X_test, y_test = pnn.nn.extract_inputs_outputs(data_test)

# Rescale y data (log, minmax)
y_train_scaled, y_test_scaled, scaler_y = pnn.nn.scale_y(y_train, y_test)
print("Rescaled data.")

# RNN: reshape data
X_train_reshaped, X_test_reshaped = pnn.nn.rnn.reshape_data(X_train, X_test)


### TRAINING
# Train multiple models and select the best one
best_model, model_metrics = pnn.nn.rnn.train_and_evaluate_models(X_train_reshaped, y_train_scaled, X_test_reshaped, y_test, scaler_y)
print("Trained model.")

# Save metrics to file
pass


### ASSESSMENT
# Apply model to test data
mean_predictions, total_variance, aleatoric_variance, epistemic_variance, std_devs = pnn.nn.rnn.predict_with_uncertainty(best_model, X_test, scaler_y)

# Save predictions to file
pass

# Sanity check
mean_metrics = pnn.nn.calculate_metrics(y_test, mean_predictions)
print("Mean prediction metrics for best-performing model:")
print(mean_metrics)
