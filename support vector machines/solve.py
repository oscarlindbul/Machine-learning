import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from SupportVectorOptimization import *
from printer import plotAll


threshold = 1e-9

def solve(datafile, kern_type="linear", kern_param=None, slack=np.inf, out="", plot=True, title=""):

    data = np.load(datafile)
    inputs = data['inputs']
    targets = data['targets']

    # precalculate matrix

    # define zerofun with targets data
    zero_func = lambda alpha : zerofun(alpha, targets)
    # define kernel function
    if kern_type == "linear":
        kernel = lin_kernel
    elif kern_type == "poly":
        kernel = lambda x, y: poly_kernel(x, y, kern_param)
    elif kern_type == "rbf":
        kernel = lambda x, y: rbf_kernel(x, y, kern_param)
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

    support_vecs = inputs[:, alpha > threshold]
    support_targets = targets[alpha > threshold]

    # calculate b = sum_i alpha_i * t_i * K(s, x_i) - t_s for SV s
    b = np.zeros(support_vecs.shape[1])
    for i in range(len(b)):
        # print("kern", alpha*kernel(support_vecs[:, i], inputs))
        b[i] = np.sum(alpha*targets*kernel(support_vecs[:, i], inputs)) - support_targets[i]
        # print("b val " + str(i), b[i])
    b = np.mean(b)

    alpha_t = alpha*targets
    indicator = lambda s : sum(alpha_t * kernel(s, inputs)) - b
    # for i in range(len(targets)):
    #     print("test " + str(i), indicator(inputs[:, i]), np.sign(indicator(inputs[:, i])) == targets[i])

    plotAll(inputs, targets, indicator)
    if not out == "":
        plt.title("{} Success: {}".format(title, solution["success"]))
        plt.savefig(out)
        plt.clf()
    if plot:
        plt.show()

    return solution
