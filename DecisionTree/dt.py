import numpy as np  
from utils.binary import BinaryClassifier
from utils import util

class DT(BinaryClassifier):
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)

        self.isLeaf = True
        self.label  = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        """
        Traverse the tree to make predictions.  You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch.
        """
        if self.isLeaf:
            return self.label
        else:
            if X[self.feature]<0.5:
                return self.left.predict(X)
            else:
                return self.right.predict(X)


    def trainDT(self, X, Y, used):
        """
        recursively build the decision tree
        """

        # get the size of the data set
        N,D = X.shape

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if self.opts['maxDepth'] <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf = True

            self.label  = util.mode(Y)
            return

        else:
            # we need to find a feature to split on
            bestFeature = -1     # which feature has lowest error
            bestError   = N      # the number of errors for this feature
            for d in range(D):
                if d in used:
                    continue

                
                leftY  = [Y[i] for i in range(N) if X[i, d] == 0]   

                rightY = [Y[i] for i in range(N) if X[i, d] == 1]


                # we'll classify the left points as their most
                # common class and ditto right points.  our error
                # is the how many are not their mode.
                leftMode = util.mode(leftY) if leftY else None
                rightMode = util.mode(rightY) if rightY else None

                leftError = sum(1 for y in leftY if y != leftMode)
                rightError = sum(1 for y in rightY if y != rightMode)

                error = leftError + rightError


                # check to see if this is a better error rate
                if error <= bestError:
                    bestFeature = d
                    bestError   = error

            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  = False   

                self.feature = bestFeature   


                self.left  = DT({'maxDepth': self.opts['maxDepth']-1})
                self.right = DT({'maxDepth': self.opts['maxDepth']-1})
                leftX = np.array([X[i] for i in range(N) if X[i, bestFeature] == 0])
                leftY = [Y[i] for i in range(N) if X[i, bestFeature] == 0]

                rightX = np.array([X[i] for i in range(N) if X[i, bestFeature] == 1])
                rightY = [Y[i] for i in range(N) if X[i, bestFeature] == 1]

    
                newUsed = used + [bestFeature]

                self.left.trainDT(leftX, leftY, newUsed)
                self.right.trainDT(rightX, rightY, newUsed)
                

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.
        """

        self.trainDT(X, Y, [])


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure.
        """

        return self
