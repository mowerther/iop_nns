"""
Script for loading data and training a neural network.
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
parser.add_argument("nn_type", help="PNN architecture to use")
parser.add_argument("-p", "--prisma", help="use PRISMA data", action="store_true")
parser.add_argument("-c", "--recalibrate", help="apply recalibration", action="store_true")
args = parser.parse_args()

# Select NN class
NN = pnn.nn.select_nn(args.nn_type)

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
    tag = f"{args.nn_type}_{scenario}"
    if args.recalibrate:
        tag += "_recal"
    print(f"\n\n\n   --- Now running: {tag} ---")

    # Select Rrs values in 5 nm steps, IOP columns
    X_train, y_train = pnn.data.extract_inputs_outputs(data_train)
    X_test, y_test = pnn.data.extract_inputs_outputs(data_test)

    # Rescale y data (log, minmax)
    y_train_scaled, y_test_scaled, scaler_y = pnn.data.scale_y(y_train, y_test)
    print("Rescaled data.")

    ### TRAINING
    # Train multiple models and select the best one
    best_model, model_metrics = pnn.nn.train_and_evaluate_models(NN, X_train, y_train_scaled, X_test, y_test, scaler_y)
    print("Trained models.")

    # Optional: Train recalibration
    if args.recalibrate:
        X_cal, y_cal = pnn.data.extract_inputs_outputs(data_cal)
        best_model = pnn.nn.recalibrate_pnn(best_model, X_cal, y_cal, scaler_y)

    # Save model to file
    saveto_model = pnn.model_path/f"{tag}_best.keras"
    best_model.save(saveto_model)
    print(f"Best model saved to {saveto_model.absolute()}")

    # Sanity check
    mdsa_all = model_metrics[["MdSA"]].unstack()
    print()
    print("MdSA values for all models:")
    print(mdsa_all.to_string())

    # Save metrics to file
    saveto_metrics = pnn.pred_path/f"{tag}_metrics_10_networks.csv"
    model_metrics.to_csv(saveto_metrics)
    print()
    print(f"All model metrics saved to {saveto_metrics.absolute()}")


    ### ASSESSMENT
    # Apply model to test data
    mean_predictions, total_variance, aleatoric_variance, epistemic_variance = best_model.predict_with_uncertainty(X_test, scaler_y)

    # Save predictions to file
    saveto_preds = pnn.pred_path/f"{tag}_preds.csv"
    pnn.modeloutput.save_model_outputs(y_test, mean_predictions, total_variance, aleatoric_variance, epistemic_variance, saveto_preds)
    print()
    print(f"Best model predictions saved to {saveto_preds.absolute()}")

    # Sanity check
    mean_metrics = pnn.nn.calculate_metrics(y_test, mean_predictions, total_variance)
    print()
    print("Mean prediction metrics for best-performing model:")
    print(mean_metrics)
    print("(Note that these may differ from the overall table due to dropout randomisation)")

    pnn.nn.scatterplot(y_test, mean_predictions, title=tag)
    pnn.nn.uncertainty_histogram(mean_predictions, total_variance, aleatoric_variance, epistemic_variance, title=tag)
