import sys
# locate other code files
sys.path.append("../..")

import numpy as np
from solve import *
from gen_data import make_data
from printer import create_anim

name = "c_poly"
y_coords = np.linspace(-3, 3, 20)
var_vals = np.linspace(0.1, 2, 5)
params = np.arange(2, 10)
seed = 5

counter = 0
K = len(y_coords)*len(var_vals)*len(params)
for k in range(len(params)):
    anim_files = [""]*len(y_coords)*len(var_vals)
    for i in range(len(var_vals)):
        for j in range(len(y_coords)):
            Ns = [10, 10, 20]
            points = np.array([(-2, 0), (2, 0), (0, y_coords[j])])
            vars = np.array([0.3, 0.3, var_vals[i]])
            classes = np.array([1, 1, -1])

            make_data(Ns, points, vars, classes, out="test", seed=seed)
            anim_file = "{}{}.jpg".format(name, counter)
            title = "Var={:0.3f}, param={}".format(var_vals[i], params[k])
            solve("test.npz", kern_type="poly", kern_param=params[k], out=anim_file, plot=False, title=title)
            anim_files[j] = anim_file
            counter += 1
            print("image {}/{}".format(counter, K))
    # print("creating animation")
    # create_anim(anim_files, out="anim_p{}.mp4".format(params[k]), duration=0.4)
