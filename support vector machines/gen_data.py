import numpy as np
import sys
import subprocess
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt

def make_data(Ns, points, vars, classes, out="data", seed=[]):
    if not seed == []:
        np.random.seed(seed)
    N = len(classes)
    num = sum(Ns)

    cluster_data = np.zeros((num, 2))
    class_data = np.ones(num)

    ind = 0
    for i in range(N):
        M = Ns[i]
        start = ind
        end = ind + M
        cluster_data[start:end, :] = read_cluster(M, points[i], vars[i])
        class_data[start:end] *= classes[i]
        ind += M

    cluster_data = np.transpose(np.array(cluster_data))

    permute = list(range(cluster_data.shape[1]))
    np.random.shuffle(permute)
    inputs = cluster_data[:, permute]
    targets = class_data[permute]
    np.savez(out, inputs=inputs, targets=targets)

def read_cluster(N, point, dev):
    return np.random.randn(N, 2) * dev + point

def check_ind(i):
    if i < 0 or i >= len(sys.argv):
        print("How to use:\n", "gen_data [option] [arg1] ([arg2] [arg3])\n\n",
        "Options:\n", "-s [seed]\n", "-o [name of output]\n", "-p [N] [(x,y)] [sigma] [class]")
        exit()

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("No arguments")
        check_ind(-1)

    i = 1
    Ns = []
    points = []
    vars = []
    classes = []
    seed = []
    name = "data"

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
            Ns.append(int(sys.argv[i-3]))
            points.append(make_tuple(sys.argv[i-2]))
            vars.append(float(sys.argv[i-1]))
            classes.append(int(sys.argv[i]))

        elif sys.argv[i] == "-o":
            i += 1
            check_ind(i)
            name = sys.argv[i]
        else:
            print("help", sys.argv[i])
            check_ind(-1)
        i += 1

    make_data(Ns, points, vars, classes, seed=seed, out=name)
