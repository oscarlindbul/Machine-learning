import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import matplotlib.pyplot as plt
import dtree
import drawtree_qt5
import pandas

ass7data = np.load("ass7_data_old.npy")
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
    print(repr("Monk{}".format(data_ind+1)).rjust(1))
    # print(repr("Means").rjust(1), repr("Vars").rjust(2))
    # for i in range(means.shape[1]):
    #     print(repr(means[data_ind, i]).rjust(1), repr(stds[data_ind,i]).rjust(2))
    table_data = np.zeros((2, means.shape[1]))
    table_data[0, :] = means[data_ind, :]
    table_data[1, :] = stds[data_ind, :]
    print(pandas.DataFrame(table_data, ["Means", "Vars"], fractions))

exp_data = np.zeros((len(fractions), 5))
exp_data[:,0] = fractions
exp_data[:,1] = 1-means[0, :]
exp_data[:,2] = stds[0, :]
exp_data[:,3] = 1-means[2, :]
exp_data[:,4] = stds[2, :]
header = ["frac", "Monk1Mean", "Monk1STD", "Monk3Mean", "Monk3STD"]
np.savetxt("exp_data.csv", exp_data, delimiter=' ', header=" ".join(header))

plt.legend()
plt.xlabel("Validation set fraction")
plt.ylabel("Classification error (with variance)")
plt.show()
