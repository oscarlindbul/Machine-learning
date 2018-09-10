import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import dtree

training_data = np.array([data.monk1, data.monk2, data.monk3])

print("Entropy")
for i in range(len(training_data)):
    print("Monk" + str(i+1), dtree.entropy(training_data[i]))
