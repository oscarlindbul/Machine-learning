import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
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
validation_data = np.array([data.monk1, data.monk2, data.monk3])

train_set, val_set = partition(training_data[0], 0.6)
tree = dtree.buildTree(train_set, data.attributes)
pruned_tree = complete_prune(tree, val_set)

drawtree_qt5.drawTree(tree)
drawtree_qt5.drawTree(pruned_tree)
