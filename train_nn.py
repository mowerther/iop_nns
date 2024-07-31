"""
Script for loading data and training a probabilistic neural network.
Trains N networks, evaluates them, and saves their outputs.

Selects the type of network from the first argument: [bnn_dc, bnn_mcd, ens_nn, mdn, rnn]
Example:
    python train_nn.py bnn_mcd

Optionally, use the -p flag to use PRISMA data:
Example:
    python train_nn.py bnn_mcd -p

Optionally, use the -c flag to enable recalibration.
Example:
    python train_nn.py bnn_mcd -c
"""
import pnn

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Script for loading data and training a neural network.")
parser.add_argument("pnn_type", help="PNN architecture to use")
parser.add_argument("-p", "--prisma", help="use PRISMA data", action="store_true")
parser.add_argument("-c", "--recalibrate", help="apply recalibration", action="store_true")
parser.add_argument("-n", "--n_models", help="number of models to train per scenario (default: 10)", type=int, default=10)
args = parser.parse_args()

# Select PNN class
PNN = pnn.nn.select_nn(args.pnn_type)

### LOAD DATA
# Load from file
scenarios, load_data = pnn.data.select_scenarios(prisma=args.prisma)
train_sets, test_sets = load_data()
print("Loaded data.")

# Split data if recalibrating
if args.recalibrate:
    train_sets, calibration_sets = pnn.recalibration.split(train_sets)
else:
    calibration_sets = [None] * len(train_sets)

# Loop over different data-split scenarios
for scenario, data_train, data_test, data_cal in zip(scenarios, train_sets, test_sets, calibration_sets):
    tag = f"{args.pnn_type}_{scenario}"
    if args.recalibrate:
        tag += "_recal"

    # Set up save folders
    saveto_model = pnn.model_path/f"{tag}.keras"
    saveto_estimates = pnn.model_estimates_path/f"{tag}_estimates.csv"
    saveto_metrics = pnn.model_estimates_path/f"{tag}_metrics.csv"

    print(f"\n\n\n   --- Now running: {tag} ---")
    print(f"Models will be saved to {saveto_model.absolute() / 'x/'}")
    print(f"Metrics will be saved to {saveto_metrics.absolute()}")

    # Select Rrs values in 5 nm steps, IOP columns
    X_train, y_train = pnn.data.extract_inputs_outputs(data_train)
    X_test, y_test = pnn.data.extract_inputs_outputs(data_test)

    # Rescale y data (log, minmax)
    y_train_scaled, y_test_scaled, scaler_y = pnn.data.scale_y(y_train, y_test)
    print("Rescaled data.")

    ### TRAINING
    # Train multiple models
    models = pnn.nn.train_N_models(PNN, X_train, y_train_scaled, n_models=args.n_models)
    print(f"Trained {args.n_models} models.")

    # Optional: Train recalibration
    if args.recalibrate:
        X_cal, y_cal = pnn.data.extract_inputs_outputs(data_cal)
        models = pnn.nn.recalibrate_pnn(models, X_cal, y_cal, scaler_y)

    # Save models to file
    pnn.nn.save_models(models, saveto_model)
    print(f"Models saved to {saveto_model.absolute()} and subfolders")

    ### ASSESSMENT
    # Calculate estimates
    estimates = pnn.nn.estimate_N_models(models, X_test, scaler_y)

    # Save estimates to file
    pnn.nn.save_estimates(y_test, estimates, saveto_estimates)
    print(f"Model predictions saved to {saveto_estimates.absolute()} and subfolders")

    # Calculate metrics
    model_metrics = pnn.nn.calculate_N_metrics(y_test, estimates)

    # Sanity check
    mdsa_all = model_metrics[["MdSA"]].unstack()
    print()
    print("MdSA values for all models:")
    print(mdsa_all.to_string())

    # Save metrics to file
    model_metrics.to_csv(saveto_metrics)
    print()
    print(f"Model metrics saved to {saveto_metrics.absolute()}")

    # Find best model and plot its results
    best_index = mdsa_all["MdSA"]["aph_443"].argmin()  # Temporary - refactor later
    mean_predictions, total_variance, aleatoric_variance, epistemic_variance = estimates[best_index]

    # Sanity check
    mean_metrics = pnn.nn.calculate_metrics(y_test, mean_predictions, total_variance)
    print()
    print("Mean prediction metrics for best-performing model:")
    print(mean_metrics)

    pnn.nn.scatterplot(y_test, mean_predictions, title=tag)
    pnn.nn.uncertainty_histogram(mean_predictions, total_variance, aleatoric_variance, epistemic_variance, title=tag)
