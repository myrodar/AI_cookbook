import numpy as np

from utils.binary import BinaryClassifier
from utils import util


class Perceptron(BinaryClassifier):
    """
    This class defines the perceptron implementation of a binary
    classifier.  See binary.py for details on the abstract class that
    this implements.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  You probably need to (at
        least) keep track of a weight vector and a bias.  We'll just
        call the 'reset' function to do this for us.
        We will also want to compute simple statistics about how the
        training of the perceptron is going.  In particular, you
        should keep track of how many updates have been made total.
        """

        BinaryClassifier.__init__(self, opts)
        self.opts = opts
        self.reset()

    def reset(self):
        """
        Reset the internal state of the classifier.
        """

        self.weights = 0          # our weight vector (lazy init)
        self.bias    = 0.0        # our bias
        self.numUpd  = 0          # number of updates made

    def online(self):
        """
        Our perceptron is online
        """
        return True

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)   +  ", b=" + repr(self.bias)

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the margin at this point.
        Semantically, a return value <0 means class -1 and a return
        value >=0 means class +1
        """

        if self.numUpd == 0:
            return 0          # failure
        else:
            return np.dot(self.weights, X) + self.bias   # this is done for you!

    def nextExample(self, X, Y):
        """
        X is a vector training example and Y is its associated class.
        We're guaranteed that Y is either +1 or -1.  We should update
        our weight vector and bias according to the perceptron rule.
        """

        # lazily create the weight vector the first time we see data
        if isinstance(self.weights, int):
            self.weights = np.zeros_like(X, dtype=float)

        # check to see if we've made an error
        if Y * self.predict(X) <= 0:   
            self.numUpd  = self.numUpd  + 1

            # perform an update
            self.weights = self.weights + Y * X    # w ← w + y·x
            self.bias    = self.bias    + Y        # b ← b + y

    def nextIteration(self):
        """
        Indicates to us that we've made a complete pass through the
        training data.  This function doesn't need to do anything for
        the perceptron, but might be necessary for other classifiers.
        """
        return   # don't need to do anything here

    def getRepresentation(self):
        """
        Return a tuple of the form (number-of-updates, weights, bias)
        """

        return (self.numUpd, self.weights, self.bias)

    def train(self, X, Y):
        """
        (BATCH ONLY)

        X is a matrix of data points, Y is a vector of +1/-1 classes.
        """
        if self.online():
            for epoch in range(self.opts['numEpoch']):
                # loop over every data point
                for n in range(X.shape[0]):
                    # supply the example to the online learner
                    self.nextExample(X[n], Y[n])

                # tell the online learner that we're
                # done with this iteration
                self.nextIteration()
        else:
            util.raiseNotDefined()


class PermutedPerceptron(Perceptron):

    def train(self, X, Y):
        """
        (BATCH ONLY)

        X is a matrix of data points, Y is a vector of +1/-1 classes.
        """
        if self.online():
            for epoch in range(self.opts['numEpoch']):
                # loop over every data point

                
                perm = np.random.permutation(X.shape[0])   

                for n in perm:
                    # supply the example to the online learner
                    self.nextExample(X[n], Y[n])

                # tell the online learner that we're
                # done with this iteration
                self.nextIteration()
        else:
            util.raiseNotDefined()


class AveragedPerceptron(Perceptron):

    def reset(self):
        """
        Reset the internal state of the classifier.
        """

        self.numUpd  = 0    # number of updates made
        self.weights = 0    # our weight vector
        self.bias    = 0.0  # our bias

       
        self.u       = 0    # cached weights
        self.B       = 0.0  # cached bias
        self.c       = 1    # counter (timestamp)

    def nextExample(self, X, Y):
        """
        X is a vector training example and Y is its associated class.
        We're guaranteed that Y is either +1 or -1.  We should update
        our weight vector and bias according to the perceptron rule.
        """

        # lazily create the weight vector the first time we see data
        if isinstance(self.weights, int):
            self.weights = np.zeros_like(X, dtype=float)
            self.u       = np.zeros_like(X, dtype=float)

        # check to see if we've made an error
        if Y * self.predict(X) <= 0:   
            self.numUpd = self.numUpd + 1

            self.weights = self.weights + Y * X       
            self.bias    = self.bias    + Y          
            self.u       = self.u + Y * self.c * X    
            self.B       = self.B + Y * self.c       

        
        self.c = self.c + 1              

    def train(self, X, Y):
        """
        (BATCH ONLY)

        X is a matrix of data points, Y is a vector of +1/-1 classes.
        """
        if self.online():
            for epoch in range(self.opts['numEpoch']):
                # loop over every data point
                for n in range(X.shape[0]):
                    # supply the example to the online learner
                    self.nextExample(X[n], Y[n])

                # tell the online learner that we're
                # done with this iteration
                self.nextIteration()

            
            self.weights = self.weights - self.u / self.c    
            self.bias    = self.bias    - self.B / self.c    
        else:
            util.raiseNotDefined()
