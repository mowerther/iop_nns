"""
Recurrent Neural Network with Gated Recurrent Units and Monte Carlo Dropout (RNN MCD).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

from .common import nll_loss, inverse_scale_y
from ..metrics import MAD, mape, mdsa, sspb


### DATA HANDLING
def reshape_data(X_train: np.ndarray, X_test: np.ndarray, *, n_features_per_timestep: int=1) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape data for the RNN, adding a timestep axis.
    """
    # Calculate the number of wavelength steps and features
    n_samples_train, n_features = X_train.shape
    n_samples_test, _ = X_test.shape  # We already know the features count

    n_timesteps = n_features // n_features_per_timestep  # Here: One wavelength per step

    # Reshape data
    X_train_reshaped = X_train.reshape((n_samples_train, n_timesteps, n_features_per_timestep))
    X_test_reshaped = X_test.reshape((n_samples_test, n_timesteps, n_features_per_timestep))

    return X_train_reshaped, X_test_reshaped


### ARCHITECTURE
def build_rnn_mcd(input_shape: tuple, *, output_size: int=6,
                  hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="tanh") -> Model:
    """
    Construct an RNN with MCD based on the input parameters.
    To do: use functools.partial for GRU?
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Adding the first recurrent layer with input shape
    model.add(GRU(hidden_units, return_sequences=(n_layers > 1),
                  activation=activation, kernel_regularizer=l2(l2_reg)))

    # Adding additional GRU layers if n_layers > 1, with Dropout layers between them
    for i in range(1, n_layers):
        model.add(Dropout(dropout_rate))
        model.add(GRU(hidden_units, return_sequences=(i < n_layers-1),
                      activation=activation, kernel_regularizer=l2(l2_reg)))

    # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
    model.add(Dense(output_size * 2, activation='linear'))

    return model


### TRAINING
def train_rnn_mcd(model: Model, X_train: np.ndarray, y_train: np.ndarray, *,
                  epochs: int=1000, batch_size: int=512, learning_rate: float=0.001, validation_split: float=0.1) -> tuple[Model, History]:
    """
    Train an RNN Model on the provided X and y data, with early stopping.
    """
    # Setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    early_stopping = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="min", restore_best_weights=True)

    # Training
    model.compile(optimizer=optimizer, loss=nll_loss)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    return model, history


def build_and_train_rnn_mcd(X_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    Build and train an RNN model on the provided X and y data, with early stopping.
    Convenience function combining the build and train functions.
    """
    # Setup
    input_shape = X_train.shape[1:]
    output_size = y_train.shape[-1]

    model = build_rnn_mcd(input_shape, output_size=output_size)
    model, history = train_rnn_mcd(model, X_train, y_train)

    return model


### APPLICATION
def predict_with_uncertainty(model: Model, X: np.ndarray, scaler_y: MinMaxScaler, *, n_samples=100):
    """
    Use the given model to predict y values for given X values, including the rescaling back to regular units.
    """
    # Generate predictions in scaled space
    pred_samples = [model.predict(X, batch_size=32, verbose=0) for _ in range(n_samples)]
    pred_samples = np.array(pred_samples)

    N = scaler_y.n_features_in_  # Number of predicted values
    mean_predictions_scaled = pred_samples[..., :N]
    raw_variances_scaled = pred_samples[..., N:]
    variance_predictions_scaled = tf.nn.softplus(raw_variances_scaled)

    # Convert from scaled space to log space
    mean_predictions, variance_predictions = inverse_scale_y(mean_predictions_scaled, variance_predictions_scaled, scaler_y)

    # Calculate aleatoric and epistemic variance in the original space
    aleatoric_variance = np.mean(variance_predictions, axis=0)
    epistemic_variance = np.var(mean_predictions, axis=0)
    total_variance = aleatoric_variance + epistemic_variance
    std_devs = np.sqrt(total_variance)

    mean_predictions = np.mean(mean_predictions, axis=0)  # Average over n_samples

    return mean_predictions, total_variance, aleatoric_variance, epistemic_variance, std_devs

def calculate_metrics(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) and other metrics between true and predicted values.

    Args:
    - y_true: Actual values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - Tuple of metrics (obs_cor, MAPD, MAD, sspb, mdsa) for the predictions.
    """
    # Ensure y_true and y_pred are numpy arrays for element-wise operations
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    # Calculate metrics assuming these functions are vectorized and can handle arrays
    # Requires definition
    mapd_values = mape(y_true, y_pred)
    mad_values = MAD(y_true, y_pred)
    sspb_values = sspb(y_true, y_pred)
    mdsa_values = mdsa(y_true, y_pred)

    # obs_cor is a scalar, all other values should be vectors of the same length as y_true/pred has columns
    obs_cor = len(y_pred)

    return obs_cor, mapd_values, mad_values, sspb_values, mdsa_values

# Calculate metrics for each target variable and store them in a DataFrame
def calculate_and_store_metrics(y_test, mean_preds, y_columns):
    # Initialize a dictionary to hold the metrics
    metrics_dict = {'obs_cor': []}
    # Initialize keys for each metric with empty lists
    for metric_name in ['MAPD', 'MAD', 'sspb', 'mdsa']:
        metrics_dict[metric_name] = []

    # Calculate metrics for each target variable
    for i in range(y_test.shape[1]):
        y_true = y_test[:, i]
        y_pred = mean_preds[:, i]
        obs_cor, mapd, mad, sspb, mdsa = calculate_metrics(y_true, y_pred)

        # Append the scalar metrics
        metrics_dict['obs_cor'].append(obs_cor)
        metrics_dict['MAPD'].append(mapd)  # Assuming mapd is a scalar
        metrics_dict['MAD'].append(mad)    # Assuming mad is a scalar
        metrics_dict['sspb'].append(sspb)  # Assuming sspb is a scalar
        metrics_dict['mdsa'].append(mdsa)  # Assuming mdsa is a scalar

    # Create a DataFrame from the dictionary
    metrics_df = pd.DataFrame(metrics_dict, index=y_columns)

    return metrics_df

def train_and_evaluate_models(X_train, y_train_scaled, X_test, y_test, y_columns, scaler_y,
                              input_shape, num_models=5):

    all_models = []
    all_mdsa = []

    best_overall_model = None
    min_total_mdsa = float('inf')
    best_model_index = -1

    for i in range(num_models):
        # activation: relu here, but I think it should be actually tanh.
        model = build_rnn_mcd(input_shape, activation='relu')
        # Train the model
        model, history = train_rnn_mcd(model, X_train, y_train_scaled)

        print('Calculating mean_preds.')
        mean_preds, total_var, aleatoric_var, epistemic_var, std_preds = predict_with_uncertainty(model, X_test, scaler_y, n_samples=100)

        print(f'Model {i+1}/{num_models}: Completed prediction with uncertainty.')

        print('Completed predict with uncertainty.')

        # Temporary line for testing
        # return model, scaler_y, mean_preds, total_var, aleatoric_var, epistemic_var, std_preds

        print('Calculating metrics.')
        metrics_df = calculate_and_store_metrics(y_test, mean_preds, y_columns)

        all_models.append(model)
        all_mdsa.append(metrics_df['mdsa'].values)

        total_mdsa = metrics_df.loc[['aCDOM_443', 'aNAP_443', 'aph_443'], 'mdsa'].sum().sum()  # Sum the mdsa of the specified variables
        if total_mdsa < min_total_mdsa:
            min_total_mdsa = total_mdsa
            best_overall_model = model
            best_model_index = i

    mdsa_df = pd.DataFrame(all_mdsa, columns=y_columns)

    print(f'The best model index is: {best_model_index}')

    return best_overall_model, best_model_index, mdsa_df

#### Execution starts here:
if __name__ == "__main__":
    # Call - train 5 models
    best_model, best_model_index, mdsa_df = train_and_evaluate_models(X_train_reshaped, y_train_scaled, X_test, y_test, y_columns,scaler_y=scaler_y, input_shape = (n_timesteps, n_features_per_timestep), num_models=5)

    # model, scaler_y, mean_preds, total_var, aleatoric_var, epistemic_var, std_preds = train_and_evaluate_models(X_train_reshaped, y_train_scaled, X_test, y_test, y_columns,scaler_y=scaler_y, input_shape = (n_timesteps, n_features_per_timestep), num_models=1)
    # raise Exception

    # inspect: mdsa_df, mdsa_df.std() etc. - can also save the best_model or use it to make predictions for plotting etc.

    mean_preds, total_var, aleatoric_var, epistemic_var, std_preds = predict_with_uncertainty(best_model, X_test, scaler_y, n_samples=30)
    metrics_df = calculate_and_store_metrics(y_test, mean_preds, y_columns)
    print(metrics_df)

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import linregress

    # Calculate percentage uncertainties relative to the mean predictions
    percent_total_uncertainty = (np.sqrt(total_var) / mean_preds) * 100
    percent_aleatoric_uncertainty = (np.sqrt(aleatoric_var) / mean_preds) * 100
    percent_epistemic_uncertainty = (np.sqrt(epistemic_var) / mean_preds) * 100

    # Create subplots: 2 rows, 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    # Apply the mask for values greater than 10^-4
    mask = (y_test > 1e-4) & (mean_preds > 1e-4)

    # Normalize the uncertainty values for the colormap within the range [0, 200]
    norm = plt.Normalize(vmin=0, vmax=200)

    # Titles
    titles = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']

    for i, ax in enumerate(axs):

        # Apply mask to all x-y
        x_values = y_test[:, i][mask[:, i]]
        y_values = mean_preds[:, i][mask[:, i]]
        color_values = percent_total_uncertainty[:, i][mask[:, i]]  # Use the corresponding uncertainties

        # scatter
        sc = ax.scatter(x_values, y_values, c=color_values, cmap='cividis', norm=norm, alpha=0.6)

        # old lin reg
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x_values), np.log(y_values))
        x_reg = np.linspace(min(x_values), max(x_values), 500)
        y_reg = np.exp(intercept + slope*np.log(x_reg))
        ax.plot(x_reg, y_reg, color='grey', label=f'Regression (R²={r_value**2:.2f})')

        # 1:1 Line
        limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(limits, limits, ls='--', color='black')


        ax.set_title(titles[i])
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-3, 10)
        ax.set_ylim(1e-3, 10)
        ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()

    # Add a single colorbar to the right of the subplots
    cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
    cbar = plt.colorbar(mappable=sc, cax=cbar_ax, orientation='vertical', norm=norm)
    cbar.set_label('Uncertainty Percentage (%)')

    plt.show()
