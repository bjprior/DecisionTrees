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
import entropy as ent


class LeafNode(object):
    """
    A leafNode

    Attributes
    ----------
    A letter which contains the majority label

    """

    def __init__(self, letter=" "):
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

    left_set: array
        The left set of the array with data less than or equal to the threshold value of the attribute

    right_set: array
        The right set of the array with data greater than the threshold value of the attribute
    """

    def __init__(self, col, thres):
        self.split_col = col
        self.threshold = thres

    def InduceDecisionTree(self, x, y):

        total_set = np.zeros((len(x), (len(x[0]) + 1)), dtype=int)
        total_set[:, :-1] = x
        total_set[:, -1] = y

        letter = y[0]
        row = x[0]
        y_count = 1
        x_count = 1

        # Check for multiple letters in data set
        for i in range(len(y)):
            if y[i] != letter:
                y_count += 1
            if x[i].all() != row.all():
                x_count += 1


        # Check array is empty, return a null leaf Node, if only one label return letter
        if len(y) == 0:
            return LeafNode(" ")
        elif y_count == 1 or x_count == 1:
            return LeafNode(letter)
        else:
            self.split_col, self.threshold = ent.findBestNode(total_set)

            left, right = SplitDataset(self.split_col, self.threshold, total_set)

            print(left)
            print(right)

            if len(left) != 0:
                child_left_x, child_left_y = SplitXY(left)
                self.left_node = Node(0, 0)
                self.left_node = Node.InduceDecisionTree(self.left_node, child_left_x, child_left_y)
            else:
                self.left_node = LeafNode(" ")

            if len(right) != 0:
                child_right_x, child_right_y = SplitXY(right)
                self.right_node = Node(0, 0)
                self.right_node = Node.InduceDecisionTree(self.right_node, child_right_x, child_right_y)
            else:
                self.right_node = LeafNode(" ")

            return self


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

        Decision_Tree = Node(0,0)
        Decision_Tree.InduceDecisionTree(x,y)

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


def SplitDataset(attribute, value, dataset):
    left_list = list()
    right_list = list()

    for row in dataset:
        if row[attribute] <= value:
            left_list.append(row)
        else:
            right_list.append(row)

    left = np.array(left_list)
    right = np.array(right_list)

    return left, right


def SplitXY(dataset):
    columns = len(dataset[0])
    y_s = dataset[:, columns - 1]
    x_s = dataset[:, :-1]

    return x_s, y_s


if __name__ == "__main__":

    data = dr.parseFile("data/train_full.txt")
    x, y = SplitXY(data)
    Tree = DecisionTreeClassifier()
    Tree.train(x, y)
