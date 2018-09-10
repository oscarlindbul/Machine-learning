import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import dtree
import drawtree_qt5

training_data = np.array([data.monk1, data.monk2, data.monk3])
test_data = np.array([data.monk1test, data.monk2test, data.monk3test])

for i in range(len(training_data)):
    tree = dtree.buildTree(training_data[i], data.attributes)
    E_train = dtree.check(tree, training_data[i])
    E_test = dtree.check(tree, test_data[i])
    print(repr("Monk{}".format(i+1)).rjust(1), repr(E_train).rjust(2), repr(E_test).rjust(3))
