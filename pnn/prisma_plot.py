import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path_template = "C:/SwitchDrive/Data/prisma_results/case_{case}/{filename}"

# RNN results to be added
model_file_mapping = {
    'BNN MCD': 'bnn_mcd',
    'BNN DC': 'bnn_dc',
    'MDN': 'mdn',
    'ENS NN': 'ens_nn'
}

def load_mdsa_data(model, case, ac_method=None):
    model_file_name = model_file_mapping[model]
    if case == 1:
        filename = f"case_{case}_{model_file_name}.csv"
    else:
        ac_suffix = 'aco' if ac_method == 'ACOLITE' else 'l2'
        filename = f"case_{case}_{model_file_name}_{ac_suffix}.csv"
    
    filepath = path_template.format(case=case, filename=filename)
    try:
        df = pd.read_csv(filepath)
        return df['aph_443'].values, df['aph_675'].values
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return [], []

models = ['BNN MCD', 'BNN DC', 'MDN', 'ENS NN']
cases = [1, 2, 3]
ac_methods = ['L2', 'ACOLITE']

data = {
    'aph_443': {case: {model: [] for model in models} for case in cases},
    'aph_675': {case: {model: [] for model in models} for case in cases}
}

# Load data for all models and cases
for model in models:
    for case in cases:
        if case == 1:
            aph_443, aph_675 = load_mdsa_data(model, case)
            data['aph_443'][case][model] = aph_443
            data['aph_675'][case][model] = aph_675
        else:
            for ac_method in ac_methods:
                aph_443, aph_675 = load_mdsa_data(model, case, ac_method)
                data['aph_443'][case][model].extend(aph_443)
                data['aph_675'][case][model].extend(aph_675)


# plot
fig, (ax1, ax2)  = plt.subplots(2, 1, figsize=(8, 10))

colors = {'BNN MCD': '#6699CC', 'BNN DC': '#997700', 'MDN': '#994455', 'ENS NN': '#EE99AA'}

def plot_boxplots(ax, data, title):
    offset = 0
    positions = []
    labels = []

    for case in cases:
        if case == 1:
            case_data = [data[case][model] for model in models]
            case_colors = [colors[model] for model in models]
            bp = ax.boxplot(case_data, positions=np.arange(offset, offset + len(models)), widths=0.6, patch_artist=True)
            for patch, color in zip(bp['boxes'], case_colors):
                patch.set_facecolor(color)
            positions.extend(np.arange(offset, offset + len(models)))
            offset += len(models)
            labels.append('Case 1')
        else:
            for method in ['L2', 'ACOLITE']:
                case_data = [data[case][model][len(data[case][model])//2:] if method == 'ACOLITE' 
                             else data[case][model][:len(data[case][model])//2] for model in models]
                case_colors = [colors[model] for model in models]
                bp = ax.boxplot(case_data, positions=np.arange(offset, offset + len(models)), widths=0.6, patch_artist=True)
                for patch, color in zip(bp['boxes'], case_colors):
                    patch.set_facecolor(color)
                positions.extend(np.arange(offset, offset + len(models)))
                offset += len(models)
                labels.append(f'Case {case}\n{method}')
    
    ax.set_xticks([1.7, 5.5, 9.5, 13.5, 17.5])
    ax.set_xticklabels(['Case 1', 'Case 2\nL2', 'Case 2\nACOLITE', 'Case 3\nL2', 'Case 3\nACOLITE'])
    ax.set_title(title)
    ax.set_ylim(0,250)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add vertical lines to separate cases
    for position in [3.5, 7.5, 11.5, 15.5]:
        ax.axvline(x=position, color='gray', linestyle='--', alpha=0.5)

plot_boxplots(ax1, data['aph_443'], 'a$_{ph}$ 443 nm')
plot_boxplots(ax2, data['aph_675'], 'a$_{ph}$ 675 nm')

# Create a single, large legend
handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in colors.values()]
leg = fig.legend(handles, colors.keys(), loc='lower center', ncol=1, bbox_to_anchor=(1.11, 0.425),
                 fontsize=14, handlelength=3, handleheight=1.5, handletextpad=1, columnspacing=1)

fig.supylabel('MdSA [%]',y=0.5)

# Size of legend patches
for handle in leg.legendHandles:
    handle.set_width(40)
    handle.set_height(20)

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.tight_layout()
plt.show()
