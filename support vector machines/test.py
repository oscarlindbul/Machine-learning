from solve import *
from gen_data import make_data

Ns = [10, 20]
points = np.array([(1, 0.5), (2, 2)])
vars = np.array([0.5, 0.1])
classes = np.array([-1, 1])

make_data(Ns, points, vars, classes, out="test")
solve("test.npz", out="test.png", plot=False)
