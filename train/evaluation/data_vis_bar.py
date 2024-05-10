import numpy as np
import matplotlib.pyplot as plt

# Data for the bars
labels = ['Old Model', 'New Model']

# CSV file 005_01 values
# y1 = np.array([433, 470])
# y2 = np.array([42, 20])
# y3 = np.array([25, 10])

# CSV file 005_02 values
y1 = np.array([432, 465])
y2 = np.array([41, 23])
y3 = np.array([27, 12])
plt.bar(labels, y1)
plt.bar(labels, y2, bottom=y1)
plt.bar(labels, y3, bottom=y1+y2)

plt.title("Comparison of Old Model vs New Model")
plt.ylabel("Number of episodes")
plt.legend(["No. Of Successes", "Collision Failures", "Timeout Failures"])

# Show the plot
plt.savefig("Old Model vs New Model.png")
plt.show()
