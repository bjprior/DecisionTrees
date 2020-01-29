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

    def __init__(self, letter):
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

    left_node: Node
        The Node to the left, either leads to another Node or a LeafNode(label)

    right_node: Node
        The Node to the right, either leads to another Node or a LeafNode(label)
    """

    def __init__(self, col, threshold):
        self.split_col = col
        self.threshold = threshold

    def InduceDecisionTree(self, x, y):

        total_set = np.zeros((len(x), (len(x[0]) + 1)), dtype=int)
        total_set[:, :-1] = x
        total_set[:, -1] = y

        # Gets the max label in the data set
        letter = np.bincount(y).argmax()

        row = x[0]
        y_count = 1
        x_count = 1

        # Check for multiple letters and differing rows in data set
        for i in range(len(y)):
            if y[i] != letter:
                y_count += 1
            if x[i].all() != row.all():
                x_count += 1

        # Check array is empty, return a null leaf Node, if only one label return letter
        if len(y) == 0:
            return LeafNode(" ")
        # Check if array is splittable if not return max label
        elif y_count == 1 and x_count == 1:
            #print(chr(letter))
            return LeafNode(letter)
        else:
            self.split_col, self.threshold = ent.findBestNode(total_set)
            #print("Attribute: "+str(self.split_col))
            #print("Threshold: "+str(self.threshold))
            left, right = SplitDataSet(self.split_col, self.threshold, total_set)


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

        self.decision_tree = Node(0, 0)
        self.decision_tree.InduceDecisionTree(x, y)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict(self, attributeInstances):
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

        # set up empty list (will convert to numpy array)
        predictions = list()

        #predictions = np.zeros((attributeInstances.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # remember to change this if you rename the variable

        for attributeList in attributeInstances:
            predictions.append(goDownTree(self, attributeList))

        print(predictions)
        return np.asarray(predictions)



def SplitDataSet(attribute, value, data_set):
    left_list = list()
    right_list = list()

    for row in data_set:
        if row[attribute] <= value:
            left_list.append(row)
        else:
            right_list.append(row)

    left = np.array(left_list)
    right = np.array(right_list)

    return left, right


def SplitXY(data_set):
    columns = len(data_set[0])
    y_s = data_set[:, columns - 1]
    x_s = data_set[:, :-1]

    return x_s, y_s

#######
    # HELPER FUNCTION FOR TASK 2.2

def goingDownCurrentBranch(currentNode, attributeEntry): # goes down current branch - returns characteristic if leaf
    if isinstance(currentNode, LeafNode):
        return currentNode.letter
    else:
        attributePosition = currentNode.split_col
        threshold = currentNode.threshold

        if (attributeEntry[attributePosition] <= threshold):
            currentNode = currentNode.left_node
            return goingDownCurrentBranch(currentNode, attributeEntry)
        else:
            currentNode = currentNode.right_node
            return goingDownCurrentBranch(currentNode, attributeEntry)

def goDownTree(tree, attributeEntry): # goes down tree, returning chracteristic
    currentNode = tree.decision_tree
    return chr(goingDownCurrentBranch(currentNode, attributeEntry))



#####

if __name__ == "__main__":
    data = dr.parseFile("data/train_noisy.txt")
    x, y = SplitXY(data)
    Tree = DecisionTreeClassifier()
    Tree.train(x, y)
    Tree.predict(data)
