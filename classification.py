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
import eval

class LeafNode(object):
    """
    A leafNode

    Attributes
    ----------
    A letter which contains the majority label

    """

    def __init__(self, letter, leafSize):
        self.letter = letter
        self.leafSize = leafSize
        # print("LeafNode: " + str(letter))

    def prune(self):
        return False, self


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

    def __init__(self, split_col, threshold, leftData, rightData):
        # print("Node: " + str(split_col) + " " + str(threshold))
        self.split_col = split_col
        self.threshold = threshold
        self.left_node = Node.induceDecisionTree(leftData[:, :-1], leftData.T[-1])
        self.right_node = Node.induceDecisionTree(rightData[:, :-1], rightData.T[-1])

    @staticmethod
    def induceDecisionTree(attributes, classification):
        dataSet = np.array(np.c_[attributes, classification])

        attributeRepeats = len(np.unique(dataSet[:, :-1], axis=0))
        classificationRepeats = len(np.unique(dataSet[:, -1]))

        if (len(dataSet) == 1) or (attributeRepeats == 1) or (classificationRepeats == 1):
            return LeafNode(dataSet[0][-1], len(dataSet))

        split_col, threshold, leftChildData, rightChildData = ent.findBestNode(dataSet)

        return Node(split_col, threshold, leftChildData, rightChildData)

    def prune(self, decTree, accuracy, validationData):
        if isinstance(self.left_node, LeafNode) and isinstance(self.right_node, LeafNode):
            return self.compact(), accuracy, True
        elif isinstance(self.left_node, Node):
            savedNode = self.left_node
            self.left_node, accuracy, compacted = self.left_node.prune(decTree, accuracy, validationData)
            newAccuracy = eval.Evaluator.getAccuracyOfDecisionTree(decTree, validationData)
            if compacted and newAccuracy < accuracy:
                self.left_node = savedNode
            else:
                accuracy = newAccuracy
        elif isinstance(self.right_node, Node):
            savedNode = self.right_node
            self.right_node, accuracy, compacted = self.right_node.prune(decTree, accuracy, validationData)
            newAccuracy = eval.Evaluator.getAccuracyOfDecisionTree(decTree, validationData)
            if compacted and newAccuracy < accuracy:
                self.right_node = savedNode
            else:
                accuracy = newAccuracy
        print(accuracy)
        return self, accuracy, False

    @staticmethod
    def pruneChildNode(childNode, decTree, accuracy):
        savedNode = childNode
        savedAccuracy = accuracy
        accuracy, childNode = childNode.prune(decTree)
        if accuracy < savedAccuracy:
            childNode = savedNode
            accuracy = savedAccuracy
        return childNode, accuracy

    def compact(self):
        if self.left_node.leafSize > self.right_node.leafSize:
            majorityLetter = self.left_node.letter
        else:
            majorityLetter = self.right_node.letter
        return LeafNode(majorityLetter, (self.right_node.leafSize + self.left_node.leafSize))


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
        self.rootNode = None

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

        self.rootNode = Node.induceDecisionTree(x, y)

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

        # predictions = np.zeros((attributeInstances.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # remember to change this if you rename the variable

        for attributeList in attributeInstances:
            predictions.append((DecisionTreeClassifier.predictInstance(self.rootNode, attributeList)))

        # print(predictions)
        return np.asarray(predictions)

    @staticmethod
    def predictInstance(node, attributeList):
        if isinstance(node, LeafNode):
            return node.letter
        else:
            if int(attributeList[node.split_col]) <= int(node.threshold):
                return DecisionTreeClassifier.predictInstance(node.left_node, attributeList)
            else:
                return DecisionTreeClassifier.predictInstance(node.right_node, attributeList)

    def prune(self, validationFname):
        validationData = dr.parseFile("data/validation.txt")
        accuracy = eval.Evaluator.getAccuracyOfDecisionTree(self, validationData)
        self.rootNode.prune(self, accuracy, validationData)


if __name__ == "__main__":
    data = dr.parseFile("data/train_full.txt")
    x, y = data[:, :-1], data.T[-1]
    # print(y)
    tree = DecisionTreeClassifier()
    tree.train(x, y)
    tree.predict(data)
    tree.prune("data/validation.txt")
