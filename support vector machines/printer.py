import numpy as np
import matplotlib.pyplot as plt
import imageio



def plotAll(X, t, indicator):
    plotPoints(X, t)
    min = np.min(np.min(X, axis=1))
    max = np.max(np.max(X, axis=1))
    plotContour(indicator, min, max)
    plt.axis('equal')


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


def plotContour(indicator, min, max):
    xgrid = np.linspace(min, max)
    ygrid = np.linspace(min, max)

    grid = np.array([[indicator(np.array([x, y])) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'), linewidth=(1, 3, 1))

def create_anim(anim_files, out = "anim.mpf", duration=0.1):
    with imageio.get_writer(out) as writer:
        for i in range(len(anim_files)):
            writer.append_data(imageio.imread(anim_files[i]))
