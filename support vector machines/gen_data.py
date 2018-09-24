import numpy as np
import sys
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt

def read_cluster(N, point, dev):
    return np.random.randn(N, 2) * dev + point

def check_ind(i):
    if i < 0 or i >= len(sys.argv):
        print("How to use:\n", "gen_data [option] [arg1] ([arg2] [arg3])\n\n",
        "Options:\n", "-s [seed]\n", "-o [name of output]\n", "-p [N] [(x,y)] [sigma] [class]")
        exit()

name = "data"
i = 1
cluster_data = []
class_data = []

while i < len(sys.argv):
    if sys.argv[i] == "-s":
        i += 1
        check_ind(i)
        seed = int(sys.argv[i])
        np.random.seed(seed)
    elif sys.argv[i] == "-p":
        #input = N (x, y) deviation
        i += 4
        check_ind(i)
        N = int(sys.argv[i-3])
        point = make_tuple(sys.argv[i-2])
        dev = float(sys.argv[i-1])
        c = int(sys.argv[i])
        data = read_cluster(N, point, dev)
        c_data = np.ones(data.shape[0])*c
        if len(cluster_data) > 0:
            cluster_data = np.concatenate((cluster_data, data))
            class_data = np.concatenate((class_data, c_data))
        else:
            cluster_data = read_cluster(N, point, dev)
            class_data = c_data
    elif sys.argv[i] == "-o":
        i += 1
        check_ind(i)
        name = sys.argv[i]
    else:
        print("help", sys.argv[i])
        check_ind(-1)
    i += 1

permute = list(range(len(cluster_data)))
np.random.shuffle(permute)
inputs = cluster_data[permute, :]
targets = class_data[permute]
np.savez(name, inputs=inputs, targets=targets)
