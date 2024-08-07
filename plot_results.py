import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import os
import re
import copy
from decimal import Decimal, getcontext
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter

params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': r'\usepackage{fontspec,physics}',
}

mpl.rcParams.update(params)

INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #for algos titles/labels

TITLES = [
    'bv_n70',
    'cat_n65',
    'dnn_n51',
    'ghz_n78',
    'ising_n98',
    'knn_n129',
    'qft_n63',
    'qugan_n111',
    'swap_test_n115',
    'tfim_n128',
    'wstate_n76'
]

LABELS = {
    'bv_n70' : 'BV',
    'cat_n65' : 'CAT',
    'dnn_n51' : 'DNN',
    'ghz_n78' : 'GHZ',
    'ising_n98' : 'ISN',
    'knn_n129' : 'KNN',
    'qft_n63' : 'QFT',
    'qugan_n111' : 'QGAN',
    'swap_test_n115' : 'SWP',
    'tfim_n128' : 'TFIM',
    'wstate_n76' : 'WST'
         }

METHOD = ['TrapChange',
          'OneCache',
          'DegreeSplit',
          'PachinQo'
         ]
def load_data_from_directory(directory):
    """Load all .pkl files from a specified directory and return a dictionary of the data."""
    data_dict = {}
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    for filename in pkl_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            data_dict[filename] = data
            print(f"Loaded data from {filename} in {directory}")
    return data_dict

methods_dirs = {
    'PachinQo': 'results',
    'OneCache': 'results_OneCache',
    'DegreeSplit': 'results_DegreeSplit'
    #TrapChange is Generated
}

all_data = {}

METHOD_COLORS = {
    'PachinQo': '#002045',
    'OneCache': '#FF8025',
    'DegreeSplit': '#00E1B4',
    'TrapChange': '#FCD063'
}

for method, directory in methods_dirs.items():
    method_data = load_data_from_directory(directory)
    all_data[method] = method_data
    print(f"Data for {method} loaded and stored.")

pachinqo_data =all_data['PachinQo']
pachinqo_data_copy = copy.deepcopy(all_data['PachinQo'])

def add_trap_changes(data):
    trap_change_data = {}
    
    # Iterate through each file in the original data
    for filename, contents in data.items():
        all_layers = contents[0]
        total_moved_dist = contents[1]
        # Initialize trap changes count
        total_trap_changes = 0
        # Process each layer
        for layer in all_layers:
            eg, swaps, u3s, slm_swaps, layer_dist_moved = layer
            # Calculate trap changes
            num_trap_changes = 1 if slm_swaps else 0 
            num_trap_changes += len(swaps) // 3 
            total_trap_changes += num_trap_changes

        comp_time = contents[3] #compilation time
        trap_change_data[filename] = [all_layers, total_moved_dist, total_trap_changes+6, comp_time]
    
    return trap_change_data

def filter_and_modify_layers(data):
    modified_data = {}
    
    for filename, contents in data.items():
        all_layers = contents[0]
        total_moved_dist = contents[1]
        comp_time = contents[3] 

        filtered_layers = []
        
        for layer in all_layers:
            eg, swaps, u3s, slm_swaps, layer_dist_moved = layer
            
            # Check if eg or u3s have data, indicating the layer should be kept
            if eg or u3s:
                swaps = []
                slm_swaps = []
                filtered_layers.append([eg, swaps, u3s, slm_swaps, layer_dist_moved])
        
        # Store the updated information in the new dataset
        modified_data[filename] = [filtered_layers, total_moved_dist, contents[2], comp_time]
    
    return modified_data

trap_change_data = add_trap_changes(pachinqo_data_copy)
trap_change_data = filter_and_modify_layers(trap_change_data)
all_data['TrapChange'] = trap_change_data

U3_TIME = 2   # Time for U3 gates
CZ_TIME = 0.8    # Time for CZ gates
SWAP_TIME = 3 * CZ_TIME  # Time for swaps
SPEED = 55 #um/us
TRAP_TIME = 125 #us

CZ_ERR = 0.0048
U3_ERR = 0.000127
T1 = 4.0 * 1000000 #T1 decoherence time in us
T2 = 1.49 * 1000000 #T2 decoherence time in us


def calculate_runtime(data):
    for filename, contents in data.items():
        all_layers = contents[0]
        total_moved_dist = contents[1]
        num_trap_changes = contents[2]
        total_runtime = 0.0
        
        for layer in all_layers:
            eg, swaps, u3s, slm_swaps, layer_dist_moved = layer

            times = []
            if u3s:
                times.append(U3_TIME)
            if eg or swaps:
                times.append(CZ_TIME)
            if slm_swaps:
                times.append(SWAP_TIME)

            if times:
                layer_runtime = max(times)
            else:
                layer_runtime = 0 

            total_runtime += layer_runtime

        if total_moved_dist and SPEED > 0:
            move_time = total_moved_dist / SPEED
            total_runtime += move_time

        trap_change_time = num_trap_changes * TRAP_TIME
        total_runtime += trap_change_time
        contents.append(total_runtime)

def calculate_gate_counts(data):
    for filename, contents in data.items():
        all_layers = contents[0]
        total_u3_count = 0
        total_cz_count = 0

        for layer in all_layers:
            eg, swaps, u3s, slm_swaps, layer_dist_moved = layer

            # Count U3 gates
            total_u3_count += len(u3s)

            # Count CZ gates
            total_cz_count += len(eg)
            total_cz_count += len(swaps)
            total_cz_count += 3 * len(slm_swaps)

        # Append the computed gate counts to the file's data
        contents.append(total_u3_count)
        contents.append(total_cz_count)

def append_swap_counts(data):
    for filename, contents in data.items():
        total_swaps = 0
        total_slm_swaps = 0

        all_layers = contents[0]

        for layer in all_layers:
            eg, swaps, u3s, slm_swaps, layer_dist_moved = layer

            total_swaps += len(swaps)
            total_slm_swaps += len(slm_swaps)

        combined_count = (total_swaps / 3) + total_slm_swaps

        contents.append(combined_count)

def calculate_estimated_success_probability(data):
    CZ_ERR = 0.0048
    U3_ERR = 0.000127
    T1 = 4.0 * 1000000  # T1 decoherence time in microseconds
    T2 = 1.49 * 1000000  # T2 decoherence time in microseconds

    for filename, contents in data.items():
        # Extract necessary values from the contents
        total_runtime = contents[-1]  
        total_u3_count = contents[-3] 
        total_cz_count = contents[-2] 

        # Compute the decoherence factor for T1 and T2
        decoherence_factor = math.exp(-total_runtime / T1) * math.exp(-total_runtime / T2)

        # Compute the error probabilities for U3 and CZ gates
        u3_success_prob = (1 - U3_ERR) ** total_u3_count
        cz_success_prob = (1 - CZ_ERR) ** total_cz_count

        # Combine all factors to compute the estimated success probability
        esp = u3_success_prob * cz_success_prob * decoherence_factor

        # Append the ESP to the contents of the file
        contents.append(esp)
        
for method in METHOD:
    print("working on ",method)
    append_swap_counts(all_data[method])
    calculate_gate_counts(all_data[method])
    calculate_runtime(all_data[method])
    calculate_estimated_success_probability(all_data[method])

print (np.mean([all_data['PachinQo'][f"{algo}.pkl"][3]*1000 for algo in LABELS.keys()]))

#Runtime
import matplotlib.pyplot as plt
import numpy as np

n_groups = len(LABELS)
index = np.arange(n_groups+1)  
bar_width = 0.8/2

METHOD = [
          'DegreeSplit',
          'PachinQo'
         ]

fig = plt.figure(figsize=(10.0, 2.1))
fig.subplots_adjust(left=0.0852, top=0.87, right=0.997, bottom=0.14)

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

# Normalize the data for runtime
max_runtimes = [max(all_data[method][f"{algo}.pkl"][-2] for method in METHOD) for algo in LABELS.keys()]
normalized_runtimes = {
    method: [all_data[method][f"{algo}.pkl"][-2] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_runtimes)] for method in METHOD
}

avgs = []
for i, method in enumerate(METHOD):
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-2]/1000 for algo in LABELS.keys()] 
    avgs.append(np.mean(raw_runtimes))

print(avgs)

max_avg = max(avgs)

for i, method in enumerate(METHOD):
    runtimes = [normalized_runtimes[method][j] for j in range(n_groups)]
    runtimes.append(avgs[i]/max_avg*100)
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-2]/1000 for algo in LABELS.keys()] 
    raw_runtimes.append(avgs[i])
    bars = plt.bar(index + i * bar_width, runtimes, bar_width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)

    # Add text inside each bar
    for bar, value in zip(bars, raw_runtimes):
        y_pos = bar.get_height() - 2  
        col ='black'
        if method == 'PachinQo':
            col = 'white'
        plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.1f}', ha='center', va='top', color=col, rotation='vertical', fontsize=14)


plt.ylabel('Circuit Runtime\n(\% of the Worst Case)', fontsize=14)
plt.xticks(index + 0.2, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=14)
plt.ylim(0, 100)
plt.xlim(-0.4, 11.8)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.656, 1.02, 1., .102),
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=14, handletextpad=0.5)

plt.savefig('main_runtime.pdf')
plt.close()

#ESP
n_groups = len(LABELS)
index = np.arange(n_groups+1)  
bar_width = 0.8/2

METHOD = [
          'DegreeSplit',
          'PachinQo'
         ]

fig = plt.figure(figsize=(10.0, 2.1))
fig.subplots_adjust(left=0.0852, top=0.87, right=0.997, bottom=0.14)

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

# Normalize the data for ESP
max_esp = [max(all_data[method][f"{algo}.pkl"][-1] for method in METHOD) for algo in LABELS.keys()]
normalized_esp = {
    method: [all_data[method][f"{algo}.pkl"][-1] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_esp)] for method in METHOD
}

avgs = []
for i, method in enumerate(METHOD):
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-1] for algo in LABELS.keys()]
    avgs.append(np.mean(raw_runtimes))

print(avgs)

max_avg = max(avgs)

for i, method in enumerate(METHOD):
    esp_values = [normalized_esp[method][j] for j in range(n_groups)]
    esp_values.append(avgs[i]/max_avg*100)
    raw_esp_values = [all_data[method][f"{algo}.pkl"][-1] for algo in LABELS.keys()]  # Get raw ESP values
    raw_esp_values.append(avgs[i])
    bars = plt.bar(index + i * bar_width, esp_values, bar_width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)

    # Add text inside each bar
    for bar, value in zip(bars, raw_esp_values):
        if bar.get_height() > 50:
            y_pos = bar.get_height() - 2  
            col ='black'
            if method == 'PachinQo':
                col = 'white'
            plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, '{:.1e}'.format(value), ha='center', va='top', color=col, rotation='vertical', fontsize=14)
        else:
            y_pos = bar.get_height() + 2  
            col ='black'
            plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, '{:.1e}'.format(value), ha='center', va='bottom', color=col, rotation='vertical', fontsize=14)


plt.ylabel('Est. Success Prob. (ESP)\n(\% of the Best Case)', fontsize=14)
plt.xticks(index + 0.2, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=14)
plt.ylim(0, 100)
plt.xlim(-0.4, 11.8)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.656, 1.02, 1., .102),
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=14, handletextpad=0.5)

plt.savefig('main_esp.pdf')
plt.close()

n_groups = len(LABELS)
index = np.arange(n_groups+1)  
bar_width = 0.8/2

METHOD = [
          'DegreeSplit',
          'PachinQo'
         ]

fig = plt.figure(figsize=(10.0, 2.1))
fig.subplots_adjust(left=0.0852, top=0.87, right=0.997, bottom=0.14)

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

# Normalize the data for swap counts
max_swaps = [max(all_data[method][f"{algo}.pkl"][4] for method in METHOD) for algo in LABELS.keys()]  # Index 4 for swap counts
normalized_swaps = {
    method: [all_data[method][f"{algo}.pkl"][4] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_swaps)] for method in METHOD
}
avgs = []
for i, method in enumerate(METHOD):
    raw_runtimes = [all_data[method][f"{algo}.pkl"][4] for algo in LABELS.keys()]
    avgs.append(np.mean(raw_runtimes))

print(avgs)

max_avg = max(avgs)

for i, method in enumerate(METHOD):
    swap_values = [normalized_swaps[method][j] for j in range(n_groups)]
    swap_values.append(avgs[i]/max_avg*100)
    raw_esp_values = [all_data[method][f"{algo}.pkl"][4] for algo in LABELS.keys()]  # Get raw ESP values
    raw_esp_values.append(avgs[i])
    bars = plt.bar(index + i * bar_width, swap_values, bar_width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)

    # Add text inside each bar
    for bar, value in zip(bars, raw_esp_values):
        if bar.get_height() > 50:
            y_pos = bar.get_height() - 2  
            col ='black'
            if method == 'PachinQo':
                col = 'white'
            plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.0f}', ha='center', va='top', color=col, rotation='vertical', fontsize=14)
        else:
            y_pos = bar.get_height() + 2  
            col ='black'
            plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.0f}', ha='center', va='bottom', color=col, rotation='vertical', fontsize=14)

plt.ylabel('Number of SWAPs\n(\% of the Worst Case)', fontsize=14)
plt.xticks(index + 0.2, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=14)
plt.ylim(0, 100)
plt.xlim(-0.4, 11.8)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.656, 1.02, 1., .102),
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=14, handletextpad=0.5)

plt.savefig('main_swaps.pdf')
plt.close()

n_groups = len(LABELS)
index = np.arange(n_groups+1)  
bar_width = 0.8/2

METHOD = [
          'OneCache',
          'PachinQo'
         ]

fig = plt.figure(figsize=(10.0, 2.1))
fig.subplots_adjust(left=0.0852, top=0.87, right=0.997, bottom=0.14)

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

# Normalize the data for runtime
max_runtimes = [max(all_data[method][f"{algo}.pkl"][-2] for method in METHOD) for algo in LABELS.keys()]
normalized_runtimes = {
    method: [all_data[method][f"{algo}.pkl"][-2] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_runtimes)] for method in METHOD
}

avgs = []
for i, method in enumerate(METHOD):
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-2]/1000 for algo in LABELS.keys()] 
    avgs.append(np.mean(raw_runtimes))

print(avgs)

max_avg = max(avgs)

for i, method in enumerate(METHOD):
    runtimes = [normalized_runtimes[method][j] for j in range(n_groups)]
    runtimes.append(avgs[i]/max_avg*100)
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-2]/1000 for algo in LABELS.keys()] 
    raw_runtimes.append(avgs[i])
    bars = plt.bar(index + i * bar_width, runtimes, bar_width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)

    # Add text inside each bar
    for bar, value in zip(bars, raw_runtimes):
        y_pos = bar.get_height() - 2  
        col ='black'
        if method == 'PachinQo':
            col = 'white'
        plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.1f}', ha='center', va='top', color=col, rotation='vertical', fontsize=14)


plt.ylabel('Circuit Runtime\n(\% of the Worst Case)', fontsize=14)
plt.xticks(index + 0.2, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=14)
plt.ylim(0, 100)
plt.xlim(-0.4, 11.8)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.6675, 1.02, 1., .102),
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=14, handletextpad=0.5)

plt.savefig('one_runtime.pdf')
plt.close()

n_groups = len(LABELS)
index = np.arange(n_groups+1)  
bar_width = 0.8/2

METHOD = [
          'TrapChange',
          'PachinQo'
         ]

fig = plt.figure(figsize=(10.0, 2.1))
fig.subplots_adjust(left=0.0852, top=0.87, right=0.997, bottom=0.14)

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

# Normalize the data for runtime
max_runtimes = [max(all_data[method][f"{algo}.pkl"][-2] for method in METHOD) for algo in LABELS.keys()]
normalized_runtimes = {
    method: [all_data[method][f"{algo}.pkl"][-2] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_runtimes)] for method in METHOD
}

avgs = []
for i, method in enumerate(METHOD):
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-2]/1000 for algo in LABELS.keys()] 
    avgs.append(np.mean(raw_runtimes))

print(avgs)

max_avg = max(avgs)

for i, method in enumerate(METHOD):
    runtimes = [normalized_runtimes[method][j] for j in range(n_groups)]
    runtimes.append(avgs[i]/max_avg*100)
    raw_runtimes = [all_data[method][f"{algo}.pkl"][-2]/1000 for algo in LABELS.keys()] 
    raw_runtimes.append(avgs[i])
    bars = plt.bar(index + i * bar_width, runtimes, bar_width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)

    # Add text inside each bar
    for bar, value in zip(bars, raw_runtimes):
        y_pos = bar.get_height() - 2  
        col ='black'
        if method == 'PachinQo':
            col = 'white'
        plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.1f}', ha='center', va='top', color=col, rotation='vertical', fontsize=14)


plt.ylabel('Circuit Runtime\n(\% of the Worst Case)', fontsize=14)
plt.xticks(index + 0.2, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=14)
plt.ylim(0, 100)
plt.xlim(-0.4, 11.8)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.6485, 1.02, 1., .102),
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=14, handletextpad=0.5)

plt.savefig('trap_runtime.pdf')
plt.close()



n_groups = len(LABELS)
index = np.arange(n_groups+1)  
bar_width = 0.8/2

METHOD = [
          'TrapChange',
          'PachinQo'
         ]

fig = plt.figure(figsize=(10.0, 2.1))
fig.subplots_adjust(left=0.0852, top=0.87, right=0.997, bottom=0.14)

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

# Normalize the data for swap counts
max_swaps = [max(all_data[method][f"{algo}.pkl"][2] for method in METHOD) for algo in LABELS.keys()]  # Index 4 for swap counts
normalized_trap_changes = {
    method: [all_data[method][f"{algo}.pkl"][2] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_swaps)] for method in METHOD
}
avgs = []
for i, method in enumerate(METHOD):
    raw_runtimes = [all_data[method][f"{algo}.pkl"][2] for algo in LABELS.keys()]
    avgs.append(np.mean(raw_runtimes))

print(avgs)

max_avg = max(avgs)

for i, method in enumerate(METHOD):
    trap_changes = [normalized_trap_changes[method][j] for j in range(n_groups)]
    trap_changes.append(avgs[i]/max_avg*100)
    raw_esp_values = [all_data[method][f"{algo}.pkl"][2] for algo in LABELS.keys()]  # Get raw ESP values
    raw_esp_values.append(avgs[i])
    bars = plt.bar(index + i * bar_width, trap_changes, bar_width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)
    # Add text inside each bar
    for bar, value in zip(bars, raw_esp_values):
        if bar.get_height() > 50:
            y_pos = bar.get_height() - 2  
            col ='black'
            if method == 'PachinQo':
                col = 'white'
            plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.0f}', ha='center', va='top', color=col, rotation='vertical', fontsize=14)
        else:
            y_pos = bar.get_height() + 2  
            col ='black'
            plt.text(bar.get_x() + bar.get_width() / 2 + 0.02, y_pos, f'{value:.0f}', ha='center', va='bottom', color=col, rotation='vertical', fontsize=14)

plt.ylabel('Number of Trap Changes\n(\% of the Worst Case)', fontsize=14)
plt.xticks(index + 0.2, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=14)
plt.ylim(0, 100)
plt.xlim(-0.4, 11.8)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.6485, 1.02, 1., .102),
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=14, handletextpad=0.5)

plt.savefig('trap_changes.pdf')
plt.close()

exit()

#Trap Changes
n_groups = len(LABELS)
index = np.arange(n_groups) 
bar_width = 0.2  

fig, ax = plt.subplots()

# Normalize the data for trap changes
max_trap_changes = [max(all_data[method][f"{algo}.pkl"][2] for method in METHOD) for algo in LABELS.keys()] 
normalized_trap_changes = {
    method: [all_data[method][f"{algo}.pkl"][2] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_trap_changes)] for method in METHOD
}

for i, method in enumerate(METHOD):
    trap_changes = [normalized_trap_changes[method][j] for j in range(n_groups)]
    bars = plt.bar(index + i * bar_width, trap_changes, bar_width, alpha=0.8, color=METHOD_COLORS[method], label=method)
    
    # Add text inside each bar for raw trap change count
    for bar, value in zip(bars, [all_data[method][f"{algo}.pkl"][2] for algo in LABELS.keys()]):
        y_pos = bar.get_height() - 5
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{value:.0f}', ha='center', va='top', color='black', rotation='vertical', fontsize=8)



plt.ylabel('\n(\% of the Worst Case)', fontsize=12)
plt.xticks(index + 0.25, [LABELS[algo] for algo in LABELS.keys()] + [r'\textbf{Avg.}'], fontsize=12, rotation=50)
plt.ylim(0, 100)
plt.xlim(-0.35, 11.85)
plt.yticks(np.arange(0, 101, 20), fontsize=12)
plt.grid(which='both', axis='y', linestyle=':', linewidth=0.5, color='gray')

ax.legend(ncol=3, edgecolor='white', bbox_to_anchor=(0.3, 1.02, 1., .102), mode='expand',
          loc='lower left', borderaxespad=0.02, borderpad=0.2,
          fontsize=12, handletextpad=0.5)

plt.savefig('main_swaps.pdf')
plt.close()

fig, ax = plt.subplots()


max_comp_times = [max(all_data[method][f"{algo}.pkl"][3] for method in METHOD) for algo in LABELS.keys()]  # index 3 for comp time
normalized_comp_times = {
    method: [all_data[method][f"{algo}.pkl"][3] / max_val * 100 for algo, max_val in zip(LABELS.keys(), max_comp_times)] for method in METHOD
}

for i, method in enumerate(METHOD):
    comp_times = [normalized_comp_times[method][j] for j in range(n_groups)]
    bars = plt.bar(index + i * bar_width, comp_times, bar_width, alpha=0.8, color=METHOD_COLORS[method], label=method)
    
    # Add text inside each bar for raw compilation time
    for bar, value in zip(bars, [all_data[method][f"{algo}.pkl"][3] for algo in LABELS.keys()]):
        y_pos = bar.get_height() - 5
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{value:.1f}', ha='center', va='top', color='black', rotation='vertical', fontsize=8)


plt.ylabel('Compilation Time (\% of max per group)')
plt.title('Compilation Time')
plt.xticks(index + 1.5 * bar_width, [LABELS[algo] for algo in LABELS.keys()])
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 20))
plt.grid(which='both', axis='y', linestyle='-', linewidth=0.5, color='gray')
ax.set_axisbelow(True)
plt.legend()
plt.tight_layout()
plt.show()

