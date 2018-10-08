import sys
# locate other code files
sys.path.append("../../")

import numpy as np
from solve import *
from gen_data import make_data
from printer import create_anim

name = "moving_clusters"
y_coords = np.linspace(-2, 2, 5)
var_vals = np.linspace(0.1, 0.9, 3)
slacks = np.logspace(-5, 1, 3)
seed=100

counter = 0
K = len(y_coords)*len(var_vals)*len(slacks)
for i in range(len(var_vals)):
    anim_files = [""]*len(y_coords)
    for j in range(len(y_coords)):
        for c in range(len(slacks)):

            Ns = [10, 10, 20]
            points = np.array([(-2, 0), (2, 0), (0, y_coords[j])])
            vars = np.array([var_vals[i]]*3)
            classes = np.array([1, 1, -1])

            make_data(Ns, points, vars, classes, out="test", seed=100)
            anim_file = "{}{}.jpg".format(name, counter)
            solve("test.npz", out=anim_file, plot=False, title="Var: {:0.3f} Slack: {:0.3f}".format(var_vals[i], slacks[c]), slack=slacks[c])
            anim_files[j] = anim_file
            counter += 1
            print("image {}/{}".format(counter, K))
    # create_anim(anim_files, out="anim{}.gif".format(i), duration=0.5)
