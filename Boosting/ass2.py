import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
from lab3_funcs import *
import random

# Call the `testClassifier` and `plotBoundary` functions for this part.


testClassifier(BayesClassifier(), dataset='iris', split=0.7)

testClassifier(BayesClassifier(), dataset='vowel', split=0.7)

plotBoundary(BayesClassifier(), dataset='iris',split=0.7)
