import matplotlib.pyplot as plt
import json
num_iterations=5
# Define the path to the result files
path_uncertainty = './file_path'
path_emc = './file_path_emc'

# Load results from JSON files for both strategies
with open(path_uncertainty, 'r') as json_file:
    results_uncertainty = json.load(json_file)
with open(path_emc, 'r') as json_file:
    results_emc = json.load(json_file)

# Prepare data for plotting for both strategies
plot_data_uncertainty = {}
plot_data_emc = {}

# Process results from uncertainty sampling
for key, value in results_uncertainty.items():
    pool_size, iteration = map(int, key.strip('()').split(','))
    if pool_size not in plot_data_uncertainty and pool_size % 1000 == 0:
        plot_data_uncertainty[pool_size] = []
    if pool_size % 1000 == 0:
        plot_data_uncertainty[pool_size].append(value)

# Process results from EMC
for key, value in results_emc.items():
    pool_size, iteration = map(int, key.strip('()').split(','))
    if pool_size not in plot_data_emc and pool_size % 1000 == 0:
        plot_data_emc[pool_size] = []
    if pool_size % 1000 == 0:
        plot_data_emc[pool_size].append(value)

# Sort the data by pool size
sorted_pool_sizes = sorted(set(plot_data_uncertainty.keys()).intersection(plot_data_emc.keys()))

# Create the plot
plt.figure(figsize=(10, 6))

# Generate a color map to use consistent colors for the same pool sizes
color_map = plt.cm.get_cmap('viridis', len(sorted_pool_sizes))
colors = color_map(range(len(sorted_pool_sizes)))

for idx, pool_size in enumerate(sorted_pool_sizes):
    color = colors[idx]
    # Start from iteration 2 (index 1)
    plt.plot(range(2, num_iterations + 1), plot_data_uncertainty[pool_size][1:], 'o--', label=f'Uncertainty Pool Size: {pool_size}', color=color)
    plt.plot(range(2, num_iterations + 1), plot_data_emc[pool_size][1:], label=f'EMC Pool Size: {pool_size}', color=color)

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim([0.89, 1])
plt.title('Active Learning Experiment Results')
plt.legend()
plt.grid(True)
plt.savefig('mnist_pool_uncertainty_vs_emc.png')
plt.show()

