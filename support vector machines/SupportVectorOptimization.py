import numpy, random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# X = numpy.load("example.npz")["inputs"]
# t = numpy.load("example.npz")["targets"]

PMatrix = 0
#global globalAlpha = objective(ALPHA_HERE)

def objective(alpha):
    p_term = 0.5*numpy.dot(alpha, numpy.matmul(PMatrix, alpha))
    print("objective", p_term - numpy.sum(alpha))
    return p_term - numpy.sum(alpha)

def lin_kernel(x, y):
    if numpy.min(x.shape) == 1 and numpy.min(y.shape) == 1:
        # print("dot")
        return numpy.dot(x, y)
    else:
        # print("mat", x)
        return numpy.matmul(numpy.transpose(x), y)

def poly_kernel(x, y, param):
    return (lin_kernel(x, y) + 1)**p

def rbf_kernel(x, y, param):
    return numpy.exp(-lin_kernel(x-y, x-y) / (2*param**2))

def calculatePMatrix(x, t, kernel):

    # print("T shape", t.shape)
    tTerm = numpy.outer(t, t)
    print("T mat", tTerm)
    print("kern mat", kernel(x,x))
    global PMatrix
    PMatrix = tTerm * kernel(x, x)
    print("P mat", PMatrix)
    # print("Pmat", PMatrix.shape)
    # print("X", kernel(x,x))
    # print("T", tTerm)
    # print("Pmat0", tTerm * kernel(x,x))
#     if type == "linear":
#         linearKernel = numpy.matmul(numpy.transpose(X), X)
#
#         PMatrix = tTerm * linearKernel
#
#     elif type == "polynomial" and not param == None:
#
#         polyKernel = numpy.matmul(numpy.transpose(X), X) + 1
#         PMatrix = tTerm*(polyKernel**param)
#
# #    elif type == "rbf":


    # else:
    #     return "ERROR"


def zerofun(alpha, t):
    return numpy.dot(alpha, t)

# def binSolve(func, guess, threshold=1e-5):
#     print(points)
#     if func(guess) <= threshold:
#         return numpy.mean(points, axis=0)
#     mid = int(len(points) / 2)
#     val = numpy.sign(func(points[mid]))
#     if val == right:
#         return binSolve(points[:(mid+1)], func)
#     elif val == left:
#         return binSolve(points[mid:], func)
#     else:
#         return -1

# PMatrix = calculatePMatrix("linear")

# func = lambda x : -1 if x[0] < 0 else 1
# N = 10001
# points = numpy.zeros((N, 2))
# points[:, 0] = numpy.linspace(-5, 10, N)
# points[:, 1] = numpy.ones(N)
# # print(points[1])
# print(binSolve(points, func))