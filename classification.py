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
import matplotlib.pyplot as plt

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

    def __init__(self, letter, leaf_total, entropy=0):
        self.letter = letter
        self.leaf_total = leaf_total
        self.entropy = entropy

    def __str__(self):
        return chr(self.letter) + "\n" + "Tot:" + str(self.leaf_total) + "\n" + "S:" + str(round(self.entropy, 2))

    def NodeHeight(self):
        return 0

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

    def __init__(self, split_col, threshold, leftData, rightData, letters, entropy, node_total):
        # print("Node: " + str(split_col) + " " + str(threshold))
        self.split_col = split_col
        self.threshold = threshold
        self.left_node = Node.induceDecisionTree(leftData)
        self.right_node = Node.induceDecisionTree(rightData)
        self.letters = letters
        self.entropy = entropy
        self.node_total = node_total

    def __str__(self):
        return "x" + str(self.split_col) + "<" + str(self.threshold) + "\n" + "Tot:" + str(
            self.node_total) + "\n" + "E:" + str(round(self.entropy, 2)) + "\n" + str(self.letters) + "\n"

    def NodeHeight(self):
        return 1 + max(self.left_node.NodeHeight(), self.right_node.NodeHeight())

    @staticmethod
    def induceDecisionTree(dataSet):
        attributeRepeats = len(np.unique(dataSet[:, :-1], axis=0))
        classificationRepeats = len(np.unique(dataSet[:, -1]))
        node_total = len(dataSet)
        entropy = ent.calcEntropy(dataSet[:, -1])

        if (len(dataSet) == 1) or (attributeRepeats == 1) or (classificationRepeats == 1):
            return LeafNode(dataSet[0][-1], node_total, entropy)

        node_total = len(dataSet)
        (unique, counts) = np.unique(dataSet[:, -1], return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        letters = frequencies

        split_col, threshold, leftChildData, rightChildData = ent.findBestNode(dataSet)

        return Node(split_col, threshold, leftChildData, rightChildData, letters, entropy, node_total)

    def prune(self, decTree, accuracy, validationData):
        leftCompacted = False
        rightCompacted = False
        if isinstance(self.left_node, LeafNode) and isinstance(self.right_node, LeafNode) and (self != decTree.rootNode):
            return self.compact(), accuracy, True
        if isinstance(self.left_node, Node):
            savedNode = self.left_node
            savedAccuracy = accuracy
            self.left_node, accuracy, leftCompacted = self.left_node.prune(decTree, accuracy, validationData)
            if leftCompacted:
                accuracy = eval.Evaluator.getAccuracyOfDecisionTree(decTree, validationData[0], validationData[1])
                if accuracy < savedAccuracy:
                    self.left_node = savedNode
                    accuracy = savedAccuracy
                    leftCompacted = False
        if isinstance(self.right_node, Node):
            savedNode = self.right_node
            savedAccuracy = accuracy
            self.right_node, accuracy, rightCompacted = self.right_node.prune(decTree, accuracy, validationData)
            if rightCompacted:
                accuracy = eval.Evaluator.getAccuracyOfDecisionTree(decTree, validationData[0], validationData[1])
                if accuracy < savedAccuracy:
                    self.right_node = savedNode
                    accuracy = savedAccuracy
                    rightCompacted = False
        return self, accuracy, (leftCompacted | rightCompacted)

    def compact(self):
        if self.left_node.leaf_total > self.right_node.leaf_total:
            majorityLetter = self.left_node.letter
        else:
            majorityLetter = self.right_node.letter
        return LeafNode(majorityLetter, (self.right_node.leaf_total + self.left_node.leaf_total))


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
        self.rootNode = Node.induceDecisionTree(dr.mergeAttributesAndCharacteristics(x, y))

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

        return np.asarray(predictions)

    @staticmethod
    def predictInstance(node, attributeList):
        if isinstance(node, LeafNode):
            return chr(node.letter)
        else:
            if int(attributeList[node.split_col]) <= int(node.threshold):
                return DecisionTreeClassifier.predictInstance(node.left_node, attributeList)
            else:
                return DecisionTreeClassifier.predictInstance(node.right_node, attributeList)

    def prune(self, validationData):
        pruneOccurred = True
        accuracy = eval.Evaluator.getAccuracyOfDecisionTree(self, validationData[0], validationData[1])
        while pruneOccurred:
            node, accuracy, pruneOccurred = self.rootNode.prune(self, accuracy, validationData)
            print(accuracy)

    def plot_tree(self):
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        # set arbitrary window size, width (x1 to x2) and height (y1 to y2)
        x1 = 0
        x2 = 1000
        y = 100
        # midpoint of the window to plot root
        midx = (x1 + x2) / 2

        plt.figure(figsize=(5, 5))
        plt.axis('off')

        # plot root node as a rectangle
        # ha= horizonatal alignment, va= vertical alignment
        # text is plotted using coordinates midx (middle of width of screen) and y2 (top of screen)
        plt.text(midx, y, str(self.rootNode), size=7, color='green',
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='green'))
        # call helper functions on left and right node to plot the children
        # in the subwindows divided by midpoint
        # define the vertical distance between node and its children
        steps = 0
        DecisionTreeClassifier.plot_tree_helper(midx, self.rootNode.left_node, x1, midx, y - 5, steps)
        DecisionTreeClassifier.plot_tree_helper(midx, self.rootNode.right_node, midx, x2, y - 5, steps)
        plt.savefig("tree.png")
        plt.show()

    @staticmethod
    def plot_tree_helper(parentx, node, x1, x2, y, steps):
        # calculate mid point of the sub window
        midx = (x1 + x2) / 2
        # if node is a leaf, plot as a filled in box, else plot with a white background
        if isinstance(node, LeafNode):
            plt.text(midx, y, str(node), size=7, color='white', ha="center", va="center",
                     bbox=dict(facecolor='green', edgecolor='white'))
            plt.plot([parentx, midx], [y + 5, y], 'brown', linestyle=':', marker='')
            return
        else:
            plt.text(midx, y, str(node), size=7, color='green', ha="center", va="center",
                     bbox=dict(facecolor='white', edgecolor='green'))
            # Line to parent, adjusting the number '5' with line length required
            plt.plot([parentx, midx], [y + 5, y], 'brown', linestyle=':', marker='')
            # if not a leaf node, call this function recursively
            # stop recursion after four rows to ensure tree is correct size for report
            # if (steps == 3):
            #     return
            left_height = node.left_node.NodeHeight() + 1
            right_height = node.right_node.NodeHeight() + 1
            # update the weight value
            weight = left_height / (left_height + right_height)
            # allocates a larger space for child with largest height
            div_x = x1 + weight * (x2 - x1)
            DecisionTreeClassifier.plot_tree_helper(midx, node.left_node,
                                                    x1, div_x, y - 5, steps + 1)
            DecisionTreeClassifier.plot_tree_helper(midx, node.right_node,
                                                    div_x, x2, y - 5, steps + 1)


if __name__ == "__main__":
    trainingData = dr.parseFile("data/train_full.txt")
    validationData = dr.parseFile("data/validation.txt")
    testData = dr.parseFile("data/test.txt")

    tree = DecisionTreeClassifier()
    tree.train(trainingData[0], trainingData[1])
    # tree.predict(data)
    #tree.plot_tree()
    print(eval.Evaluator.getAccuracyOfDecisionTree(tree, testData[0], testData[1]))

    print("----------------PRUNE------------------------")
    tree.prune(validationData)
    #tree.plot_tree()
    print("----------------Test------------------------")
    print(eval.Evaluator.getAccuracyOfDecisionTree(tree, testData[0], testData[1]))


