import sys
# locate other code files
sys.path.append("../..")

import numpy as np
from solve import *
from gen_data import make_data
from printer import create_anim

name = "c_rbf"
y_coords = np.linspace(-2, 2, 10)
var_vals = np.linspace(0.3, 1.5, 3)
params = np.linspace(0.1, 1, 3)
slacks = np.logspace(-1, 4, 5)
seed = 5

counter = 0
K = len(y_coords)*len(var_vals)*len(params)*len(slacks)
for k in range(len(params)):
    anim_files = [""]*len(y_coords)*len(var_vals)
    for i in range(len(var_vals)):
        for j in range(len(y_coords)):
            for c in range(len(slacks)):
                Ns = [10, 10, 20]
                points = np.array([(-2, 0), (2, 0), (0, y_coords[j])])
                vars = np.array([var_vals[i]]*3)
                classes = np.array([1, 1, -1])

                make_data(Ns, points, vars, classes, out="test", seed=seed)
                anim_file = "{}{}.jpg".format(name, counter)
                title = "Var={:0.3f}, param={} slack={}".format(var_vals[i], params[k], slacks[c])
                solve("test.npz", kern_type="rbf", kern_param=params[k], out=anim_file, plot=False, title=title, slack=slacks[c])
                anim_files[j] = anim_file
                counter += 1
                print("image {}/{}".format(counter, K))
    # print("creating animation")
    # create_anim(anim_files, out="anim_p{}.mp4".format(params[k]), duration=0.4)
