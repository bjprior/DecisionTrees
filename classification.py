##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
import dataReader as dr

class LeafNode(object):
    """
    A leafNode

    Attributes
    ----------
    A letter which contains the majority label

    """

    def __init__(self, letter = " "):
        self.letter = letter


class Node(object):
    """
    A Node

    Attributes
    ----------
    split_col: int
        The column index referencing the attribute on which the data set will be split

    threshold: int
        The threshold value on which the data will be split for the attribute given

    """
    def __init__(self, col, thres):
        self.split_col = col
        self.threshold = thres

def InduceDecisionTree(x,y):

    letters = ["A","C","E","G","O","Q"]
    count = np.zeros(6)

    #Check for multiple letters in data set remaining

    for i in range(len(y)):
        for l in range(len(letters)):
            if(letters[l] == chr(y[i])):
                count[l] +=1
    count_max = 0
    index = 10

    # Get max count and index position
    for l in range(0,6):
        if count[l] > max:
            max = count[l]
            index = l

    #Check if all samples have same label also checks for 1 remaining
    if np.nonzero(count_max) == 1:
        return LeafNode(letters[index])
    else:
        Node_new = Node(FindBestNode(x,y))


def SplitDataset(attribute, value, dataset):

    left = list()
    right = list()

    for row in dataset:
        if row[attribute] <= value:
            left.append(row)
        else:
            right.append(row)

    return left,right

class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False
    
    
    def train(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        InduceDecisionTree(x,y)

        
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
        return self
    
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """
        
        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        
        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
    
        # remember to change this if you rename the variable
        return predictions
        

if __name__ == "__main__":
    data = dr.parseFile("data/toy.txt")
    xs = np.hsplit(data,4)
    y = xs[3]
    x = np.hstack((xs[0],xs[1],xs[2]))

    Tree = DecisionTreeClassifier

    Tree.__init__(Tree)
    Tree.train(Tree,x,y)
