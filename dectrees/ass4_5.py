import sys
# import python function files
sys.path.insert(0, "./python")

import monkdata as data
import numpy as np
import dtree
import drawtree_qt5

training_data = np.array([data.monk1, data.monk2, data.monk3])
atr_N = 6
depth = 2

set = training_data[0]

def makeTree(set, level, attributes):
    if level >= depth:
        return dtree.TreeLeaf(dtree.mostCommon(set))
    attr = dtree.bestAttribute(set, attributes)
    node = []
    branches = []
    for val in attr.values:
        subset = dtree.select(set, attr, val)
        attributes_left = [a for a in attributes if a != attr]
        if dtree.allPositive(subset):
            node = dtree.TreeLeaf(True)
        elif dtree.allNegative(subset):
            node = dtree.TreeLeaf(False)
        else:
            node = makeTree(subset, level+1, attributes_left)
        branches.append((val,node))
    node = dtree.TreeNode(attr, dict(branches), dtree.mostCommon(set))
    return node

root = makeTree(set, 0, data.attributes)
# drawtree_qt5.drawTree(root)

correct_tree = dtree.buildTree(set, data.attributes, 2)
drawtree_qt5.drawTree(correct_tree)
