"""
Script for loading map data and applying PNN estimation pixel-wise.
First loads and plots (atmospherically corrected) reflectance data.
Next finds and applies the average-performing model (by median MdSA) for each network-scenario combination.

Data are loaded from pnn.c.map_data_path by default, but a custom folder can be supplied using the -f flag (e.g. `python plot_maps.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

Recalibrated models are not currently supported.

Example:
    python plot_maps.py bnn_mcd
"""
import pnn

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(__doc__.splitlines()[1])
parser.add_argument("pnn_type", help="PNN architecture to use")
parser.add_argument("-f", "--folder", help="folder to load data from", type=pnn.c.Path, default=pnn.c.map_data_path)
parser.add_argument("-a", "--acolite", help="use acolite data (if False: use L2C)", action="store_true")
args = parser.parse_args()


### Load data
filenames = args.folder.glob(pnn.maps.pattern_prisma_acolite if args.acolite else pnn.maps.pattern_prisma_l2)

for filename in filenames:
    # Load reflectance
    scene = pnn.maps.load_prisma_map(filename, acolite=args.acolite)
    print(f"Read data from `{filename.absolute()}`")

    # Load RGB background image
    filename_h5 = pnn.maps.get_h5_filename(filename)
    rgb_cube = pnn.maps.load_h5_as_rgb(filename_h5)
    scene_rgb = pnn.maps.rgb_to_xarray(scene, rgb_cube)

    # Plot Rrs for reference
    pnn.maps.plot_Rrs(scene, title=filename.stem, background=scene_rgb)
    continue

    # Convert Rrs to list of spectra
    spectra, *_ = pnn.maps.map_to_spectra(scene)

    # Rescale Rrs to the same scale the models were trained on
    prisma_scenarios = pnn.data.read_prisma_data()
    model_scenario = prisma_scenarios[0]
    print(f"Model training scenario: {model_scenario.train_scenario}")

    X_scaler = model_scenario.X_scaler
    spectra_trans = X_scaler.transform(spectra)

    *_, y_train = pnn.data.extract_inputs_outputs(model_scenario.train_data)
    scaler_y, *_ = pnn.data.scale_y(y_train)

    # Set up output label
    PNN = pnn.nn.select_nn(args.pnn_type)
    scenario_for_average = pnn.c.prisma_gen_ACOLITE if args.acolite else pnn.c.prisma_gen_L2
    label = f"{filename.stem}-{PNN.name}-{scenario_for_average}"

    # Load average-performing PNN
    metrics = pnn.modeloutput.read_all_model_metrics(pnn.c.model_estimates_path, scenarios=[scenario_for_average])
    median_indices, *_ = pnn.modeloutput.select_median_metrics(metrics)
    median_index = median_indices.loc[scenario_for_average, args.pnn_type]
    print(f"Model number {median_index} is the average-performing {args.pnn_type} in the '{scenario_for_average}' scenario")

    model = pnn.nn.load_model_iteration(PNN, median_index, model_scenario.train_scenario)
    print(f"Loaded model: {model}")

    # Apply PNN
    iop_mean, iop_variance, *_ = model.predict_with_uncertainty(spectra_trans, scaler_y)

    # Convert IOPs to maps
    iop_map = pnn.maps.create_iop_map(iop_mean, iop_variance, scene)

    # Save IOP map
    pnn.maps.save_iop_map(iop_map, saveto=pnn.c.map_output_path/f"{label}_iops.nc")

    # Plot IOP maps - main output
    pnn.maps.plot_IOP_single(iop_map, background=scene_rgb, title=filename.stem, saveto=pnn.c.map_output_path/f"{label}_aph443.pdf")
    pnn.maps.plot_IOP_all(iop_map, background=scene_rgb, title=filename.stem, saveto=pnn.c.map_output_path/f"{label}_iops.pdf")
