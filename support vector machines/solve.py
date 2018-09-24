import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from SupportVectorOptimization import *
# from printer import plotAll

# def print_help():
#     print("solve.py [data file] ([option] [args] ...)")
#     print()
#     print("Options:")
#     print("Use kernel type: -t [kernel type (default 'linear')] [param]")
#     print("Set slack: -s [C (default inf)]")
#     print("set output file: -o [file name]")
#     print("Supported Kernels:\n", "'linear'\n", "'poly'\n", "'rbf'")
#
# def check_input():
#     dataname = sys.argv[1]
#
#     i = 2
#     while i < len(sys.argv):
#         if sys.argv[0]
#         i += 1

def solve(datafile, kern_type="linear", kern_param=None, slack=np.inf, out="fig.pdf", plot=True):

    data = np.load(datafile)
    inputs = data['inputs']
    targets = data['targets']
    print("input shape", inputs.shape)
    print("target shape", targets.shape)

    # precalculate matrix

    # define zerofun with targets data
    zero_func = lambda alpha : zerofun(alpha, targets)
    # define kernel function
    if kern_type == "linear":
        kernel = lin_kernel
    elif kern_type == "poly":
        kernel = lambda x, y: poly_kernel(x, y, kern_param)
    elif kern_type == "rbf":
        # TODO
        print("TODO")
    else:
        print("Error! Unsupported kernel type")
        return
    calculatePMatrix(inputs, targets, kernel)

    N = inputs.shape[1]
    start = np.zeros((1, N))
    # define alpha bounds
    if slack < np.inf:
        bounds = [(0, slack) for i in range(N)]
    else:
        bounds = [(0, None) for i in range(N)]
    # define equality constraint
    constraint = {'type' : 'eq', 'fun' : zero_func}

    # solve
    solution = minimize(objective, start, bounds=bounds, constraints=constraint)
    alpha = solution['x']
    print(solution["success"], solution["message"])
    print("zero func", zero_func(alpha))

    support_vecs = inputs[:, alpha > 0]
    support_targets = targets[alpha > 0]
    print(support_vecs.shape)
    # calculate b = sum_i alpha_i * t_i * K(s, x_i) - t_s for SV s
    # print(alpha, objective(alpha))
    b = np.zeros(support_vecs.shape[1])
    for i in range(len(b)):
        # print("kern", alpha*kernel(support_vecs[:, i], inputs))
        b[i] = np.sum(alpha*targets*kernel(support_vecs[:, i], inputs)) - support_targets[i]
        print(b[i])
    b = np.mean(b)

    alpha_t = alpha*targets
    indicator = lambda s : sum(alpha_t * kernel(s, inputs)) - b
    for i in range(len(targets)):
        print("test " + str(i), indicator(inputs[:, i]), np.sign(indicator(inputs[:, i])) == targets[i])

    # plotAll(inputs, targets, indicator)
