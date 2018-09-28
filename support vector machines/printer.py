import numpy as np
import matplotlib.pyplot as plt



def plotAll(X, t):
    plotPoints(X, t)

    print("Not Done Yet")

def plotPoints(X, t):
    (plusArray, minusArray) = separate(X, t)
    plt.scatter([p[0] for p in plusArray], [p[1] for p in plusArray], None, 'b')
    plt.scatter([p[0] for p in minusArray], [p[1] for p in minusArray], None, 'r')
    plt.axis('equal')
    plt.show()



def separate(X, t):
    tplus = []
    tminus = []
    for i in range(len(t)):
        if t[i] > 0:
            tplus.append(X[i,:])
        else:
            tminus.append(X[i, :])
    return (np.array(tplus), np.array(tminus))

plotPoints(np.array([[1, 1], [2, 2], [2, 3]]), np.array([1, -1, 1]))