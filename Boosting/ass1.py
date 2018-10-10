import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
from lab3_funcs import *
import random

# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.


X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
plotGaussian(X,labels,mu,sigma)
