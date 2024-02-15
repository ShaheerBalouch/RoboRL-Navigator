import math

import matplotlib.pyplot as plt
import numpy as np

# Read data from the text file
data = []
with open('orientation_values.txt', 'r') as file:
    for line in file:
        line = line.replace('[', '').replace(']', '')
        values = line.strip().split()
        x, y, z = map(float, values)
        data.append((x, y, z))

# Extract x, y, and z values
x_values = [entry[0] for entry in data]
y_values = [entry[1] for entry in data]
z_values = [entry[2] for entry in data]

# Define the same range for all histograms
hist_range = (-math.pi, math.pi)
bins = 100
# Create density plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(x_values, bins=bins, density=True, alpha=0.7, range=hist_range)
axs[0].set_title('Roll Values Density')
axs[0].set_ylim(0, 4)  # Set y-axis range

axs[1].hist(y_values, bins=bins, density=True, alpha=0.7, range=hist_range)
axs[1].set_title('Pitch Values Density')
axs[1].set_ylim(0, 4)  # Set y-axis range

axs[2].hist(z_values, bins=bins, density=True, alpha=0.7, range=hist_range)
axs[2].set_title('Yaw Values Density')
axs[2].set_ylim(0, 4)  # Set y-axis range

plt.tight_layout()
plt.show()
