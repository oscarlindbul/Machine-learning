import numpy, random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

X = numpy.load("example.npz")["inputs"]
t = numpy.load("example.npz")["targets"]


#global globalAlpha = objective(ALPHA_HERE)

def objective(alpha):
    alphaTerm = numpy.matmul(alpha, numpy.transpose(alpha))

    return 0.5*numpy.sum(alphaTerm * PMatrix) - numpy.sum(alpha)



def calculatePMatrix(type, param=None):

    tTerm = numpy.matmul(t, numpy.transpose(t))

    if type == "linear":
        linearKernel = numpy.matmul(numpy.transpose(X), X)

        return tTerm * linearKernel

    elif type == "polynomial":

        polyKernel = numpy.matmul(numpy.transpose(X), X) + 1
        return tTerm*(polyKernel**param)

#    elif type == "rbf":


    else:
        return "ERROR"


def zerofun(alpha, tt):
    return numpy.dot(alpha, tt)

PMatrix = calculatePMatrix("linear")


