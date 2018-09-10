import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import matplotlib.pyplot as plt
import dtree
import drawtree_qt5
import random

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata)*fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def complete_prune(tree, validation):
    better_found = True
    best_prune = tree
    best_perf = dtree.check(tree, validation)
    while better_found:
        better_found = False
        prunes = dtree.allPruned(best_prune)
        for prune_data in prunes:
            performance = dtree.check(prune_data, validation)
            if performance > best_perf:
                best_prune = prune_data
                best_perf = performance
                better_found = True
    return best_prune


training_data = np.array([data.monk1, data.monk2, data.monk3])
test_data = np.array([data.monk1test, data.monk2test, data.monk3test])

iters = 1000
fractions = np.arange(0.3, 0.9, 0.1)
N = len(fractions)
# (number of data sets, number of fractions, number of versions (validation/test), number of iterations)
perf_data = np.zeros((len(training_data), N, iters))

plt.figure()
for data_ind in [0, 2]:
    for iter in range(iters):
        for i in range(N):
            train_set, val_set = partition(training_data[data_ind], fractions[i])
            tree = dtree.buildTree(train_set, data.attributes)
            pruned_tree = complete_prune(tree, val_set)
            perf_data[data_ind, i, iter] = dtree.check(pruned_tree, test_data[data_ind])
            # perf_data[data_ind, i, 1, iter] = dtree.check(tree, test_data[data_ind])
        if iter % 10 == 0:
            print(np.sqrt(np.var(perf_data[data_ind, :, :iter], axis=1)))
            print("Monk{}, {:.0f}%".format(data_ind+1, iter/float(iters)*100))

    # means = np.mean(perf_data, axis=3)
    # stds = np.sqrt(np.var(perf_data, axis=3))
    # plt.errorbar(fractions, 1-means[data_ind, :], yerr=stds[data_ind, :, 0], label="Monk{} pruned".format(data_ind+1))
    # plt.errorbar(fractions, means[data_ind, :, 1], yerr=stds[data_ind, :, 1], label="Monk{} normal".format(data_ind+1), )

np.save("ass7_data", (fractions, perf_data))

# plt.legend()
# plt.xlabel("Validation set fraction")
# plt.ylabel("Classification error")
# plt.show()
