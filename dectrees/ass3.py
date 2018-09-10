import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import dtree

training_data = np.array([data.monk1, data.monk2, data.monk3])
atr_N = 6

print("InfoGain")
print(repr("Set").rjust(1), end="      ")
for j in range(atr_N):
    print(repr("a{}".format(j+1)).rjust(j+2), end="     ")
print()

for i in range(len(training_data)):
    print(repr("Monk{} ".format(i+1)).rjust(1), end=" ")
    for j in range(atr_N):
        print(repr("{0:0.5f}".format(
            dtree.averageGain(training_data[i],
            data.attributes[j]))).rjust(j+2), end=" ")
    print()
