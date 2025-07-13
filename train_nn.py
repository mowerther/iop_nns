"""
Script for loading data and training a probabilistic neural network.
Trains N networks, evaluates them, and saves their outputs.
Selects the type of network from the first argument: [bnn_dc, bnn_mcd, ens_nn, mdn, rnn].
Note that model files and estimates are saved to default folders and will override existing files; custom locations can be specified (-o, -e).

Example:
    python train_nn.py bnn_mcd
    python train_nn.py bnn_mcd -p
    python train_nn.py bnn_mcd -c
    python train_nn.py bnn_mcd -pc -o path/to/models/ -e path/to/estimates/ -n 10
"""
import pnn

### Parse command line arguments
parser = pnn.ArgumentParser(description=__doc__)
parser.add_argument("pnn_type", help="PNN architecture to use")
parser.add_argument("-d", "--data_folder", help="Folder to load data from.", type=pnn.c.Path, default=pnn.insitu_data_path)
parser.add_argument("-o", "--output_folder", help="Folder to save models to.", type=pnn.c.Path, default=pnn.model_path)
parser.add_argument("-e", "--estimates_folder", help="Folder to save model estimates to.", type=pnn.c.Path, default=pnn.model_estimates_path)
parser.add_argument("-p", "--prisma", help="Use PRISMA data.", action="store_true")
parser.add_argument("-c", "--recalibrate", help="Apply recalibration.", action="store_true")
parser.add_argument("-n", "--n_models", help="Number of models to train per scenario (default: 25).", type=int, default=25)
args = parser.parse_args()

# Select PNN class
PNN = pnn.nn.select_nn(args.pnn_type)

### LOAD DATA
# Load from file
label, *_, load_data = pnn.data.select_scenarios(prisma=args.prisma)
datascenarios = load_data(args.data_folder)
print("Loaded data.")


# Loop over different data-split scenarios
for scenario_train, data_train, scenarios_and_data_test in datascenarios:
    ### SETUP
    tag_train = f"{args.pnn_type}_{scenario_train}"
    if args.recalibrate:
        tag_train += "_recal"

    saveto_model = args.output_folder/f"{tag_train}.zip"
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

    # Train X and y rescalers
    scaler_X = pnn.data.generate_rescaler_rrs(X_train)
    scaler_y = pnn.data.generate_rescaler_iops(y_train)
    print("Trained X and y rescalers.")

    ### TRAINING
    # Train multiple models
    models = pnn.nn.train_N_models(PNN, X_train, y_train, scaler_X=scaler_X, scaler_y=scaler_y, n_models=args.n_models)
    print(f"Trained {len(models)}/{args.n_models} models.")

    # Optional: Train recalibration
    if args.recalibrate:
        X_cal, y_cal = pnn.data.extract_inputs_outputs(data_cal)
        models = pnn.nn.recalibrate_pnn(models, X_cal, y_cal)

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

        saveto_estimates = args.estimates_folder/f"{tag_test}_estimates.csv"
        saveto_metrics = args.estimates_folder/f"{tag_test}_metrics.csv"

        print("\n----------")
        print(f"Now testing: {scenario_test.label}")
        print(f"Metrics will be saved to {saveto_metrics.absolute()}")
        print("----------")

        # Pre-process data
        X_test, y_test = pnn.data.extract_inputs_outputs(data_test)

        # Calculate estimates
        estimates = pnn.nn.estimate_N_models(models, X_test)

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
