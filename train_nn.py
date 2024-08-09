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
label, *_, load_data = pnn.data.select_scenarios(prisma=args.prisma)
datascenarios = load_data()
print("Loaded data.")


# Loop over different data-split scenarios
for scenario_train, data_train, scenarios_and_data_test in datascenarios:
    ### SETUP
    tag_train = f"{args.pnn_type}_{scenario_train}"
    saveto_model = pnn.model_path/f"{tag_train}.keras"
    print("\n\n----------")
    print(f"Now training: {scenario_train.label}")
    print(f"Models will be saved to {saveto_model.absolute()}")
    print(f"Number of testing scenarios: {len(scenarios_and_data_test)}")
    print("----------\n\n")

    # Set up recalibration
    if args.recalibrate:
        data_train, data_cal = pnn.recalibration.split(data_train)
        print(f"Split training data into training ({len(data_train)}) and recalibration ({len(data_cal)}) sets.")

    # Select Rrs values in 5 nm steps, IOP columns
    X_train, y_train = pnn.data.extract_inputs_outputs(data_train)

    # Rescale y data (log, minmax)
    scaler_y, y_train_scaled = pnn.data.scale_y(y_train)
    print("Rescaled data.")


    ### TRAINING
    # Train multiple models
    models = pnn.nn.train_N_models(PNN, X_train, y_train_scaled, n_models=args.n_models)
    print(f"Trained {len(models)}/{args.n_models} models.")

    # Optional: Train recalibration
    if args.recalibrate:
        X_cal, y_cal = pnn.data.extract_inputs_outputs(data_cal)
        models = pnn.nn.recalibrate_pnn(models, X_cal, y_cal, scaler_y)

    # Save models to file
    pnn.nn.save_models(models, saveto_model)
    print(f"Models saved to {saveto_model.absolute()} in subfolders 0--{len(models)-1}.")

    ### ASSESSMENT
    ## Loop over assessment scenarios
    for scenario_test, data_test in scenarios_and_data_test.items():
        # Set up save folders
        tag_test = f"{args.pnn_type}_{scenario_test}"
        if args.recalibrate:
            tag_test += "_recal"

        saveto_estimates = pnn.model_estimates_path/f"{tag_test}_estimates.csv"
        saveto_metrics = pnn.model_estimates_path/f"{tag_test}_metrics.csv"

        print("\n----------")
        print(f"Now testing: {scenario_test.label}")
        print(f"Metrics will be saved to {saveto_metrics.absolute()}")
        print("----------")

        # Pre-process data
        X_test, y_test = pnn.data.extract_inputs_outputs(data_test)


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

        pnn.nn.scatterplot(y_test, mean_predictions, title=tag_test)
        pnn.nn.uncertainty_histogram(mean_predictions, total_variance, aleatoric_variance, epistemic_variance, title=tag_test)
