import numpy as np
import matplotlib.pyplot as plt



def plotAll(X, t, indicator):
    plotPoints(X, t)
    plotContour(indicator)
    plt.axis('equal')
    plt.show()


def plotPoints(X, t):
    (plusArray, minusArray) = separate(X, t)
    plt.scatter([p[0] for p in plusArray], [p[1] for p in plusArray], None, 'b')
    plt.scatter([p[0] for p in minusArray], [p[1] for p in minusArray], None, 'r')


def separate(X, t):
    tplus = []
    tminus = []
    for i in range(len(t)):
        if t[i] > 0:
            tplus.append(X[:,i])
        else:
            tminus.append(X[:, i])
    return (np.array(tplus), np.array(tminus))


def plotContour(indicator):
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator(np.array([x, y])) for x in xgrid] for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'), linewidth=(1, 3, 1))
