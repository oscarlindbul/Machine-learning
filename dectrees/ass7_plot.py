import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import matplotlib.pyplot as plt
import dtree
import drawtree_qt5
import random

ass7data = np.load("ass7_data.npy")
fractions = ass7data[0]
perf_data = ass7data[1]

means = np.mean(perf_data, axis=2)
stds = np.sqrt(np.var(perf_data, axis=2))
plt.figure()
for data_ind in [0, 2]:
    plt.errorbar(fractions, 1-means[data_ind, :],
            yerr=stds[data_ind, :],
            label="Monk{} pruned".format(data_ind+1),
            marker=".")

plt.legend()
plt.xlabel("Validation set fraction")
plt.ylabel("Classification error")
plt.show()
