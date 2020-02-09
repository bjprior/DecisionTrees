##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):

    @staticmethod
    def get_accuracy_of_decision_tree(decision_tree, attributes, annotation):
        """
        Computes accuracy of decision tree
        Parameters
        ----------
        decision_tree : DecisionTreeClassifier
            a DecisionTreeClassifier object
        attributes : np.array
            an N x M dimensional numpy array containing the attributes of the dataset
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels

        Returns
        -------
        int
            the corresponding accuracy of the dataset
        """

        predictions = decision_tree.predict(attributes)
        confusion_matrix = Evaluator.confusion_matrix(predictions, annotation)
        return Evaluator.accuracy(confusion_matrix)

    @staticmethod
    def confusion_matrix(prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        # iterate through rows and columns of the confusion matrix
        row = 0
        col = 0
        # storing count of when true letter is equal to some predicted letter
        for trueLetter in class_labels:
            for predictedLetter in class_labels:
                counter = 0
                for index in range(np.size(prediction)):
                    if trueLetter == annotation[index] and predictedLetter == prediction[index]:
                        counter += 1
                    confusion[row][col] = counter
                row += 1
                row %= len(class_labels)
            col += 1
            col %= len(class_labels)

        return confusion

    @staticmethod
    def accuracy(confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        # accuracy is given by instanceWhen(TRUTH == PREDICTED) / ALL EVENTS

        true_positive = np.trace(confusion)
        all_events = np.sum(confusion)

        if true_positive == 0 or all_events == 0:
            return 0
        else:
            return true_positive / all_events

    @staticmethod
    def precision(confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # precision (per characteristic) == TRUTH (trace part) / TOTAL PREDICTION THAT LETTER (row)

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        # iterate through each row of the confusion matrix
        # finding precision for ach ground truth according to equation above
        index = 0

        for letterIndex in range(np.size(confusion[:, -1])):
            if np.sum(confusion[:, letterIndex]) == 0:
                p[index] = 0
            else:
                p[index] = confusion[letterIndex][letterIndex] / np.sum(confusion[letterIndex])
            index += 1

        return p, np.average(p)

    @staticmethod
    def recall(confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        # recall (per characteristic) == TRUTH (trace part) / TOTAL TIMES THAT WAS THE TRUE LETTER (column)

        # iterate through each row of the confusion matrix
        # finding recall for each ground truth according to equation above
        index = 0

        for letterIndex in range(np.size(confusion[:, -1])):
            if (np.sum(confusion[letterIndex]) == 0):
                r[index] = 0
            else:
                r[index] = confusion[letterIndex][letterIndex] / np.sum(confusion[:, letterIndex])
            index += 1

        return r, np.average(r)

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        # f1 (per characteristic) == 2 * (PRECISION * RECALL) / (PRECISION + RECALL)

        precision, macro_p = self.precision(confusion)
        recall, macro_r = self.recall(confusion)

        # iterate through each row of the confusion matrix
        # finding f1 for each ground truth according to equation above
        index = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            f[index] = 2 * (precision[index] * recall[index]) / (recall[index] + precision[index])
            index += 1

        return f, np.average(f)


if __name__ == "__main__":
    data = dr.parseFile("data/train_full.txt")
    print("RESULTS FOR TRAIN_FULL.TXT:")
    x, y = data[:, :-1], data.T[-1]
    tree = cp.DecisionTreeClassifier()
    tree.train(x, y)
    test = dr.parseFile("data/test.txt")
    xtruth, ytruth = test[:, :-1], test.T[-1]
    predictions = tree.predict(test)
    e = Evaluator()
    a = e.confusion_matrix(ytruth, predictions)
    print("Order of matrix is ACEGOQ")
    print("confusion" + "\n" + str(a))
    print("accuracy: " + str(e.accuracy(a)))
    print("Recall: " + str(e.recall(a)))
    print("Precision: " + str(e.precision(a)))
    print("F1score: " + str(e.f1_score(a)))

    data = dr.parseFile("data/train_noisy.txt")
    print("RESULTS FOR TRAIN_NOISY.TXT:")
    x, y = data[:, :-1], data.T[-1]
    tree = cp.DecisionTreeClassifier()
    tree.train(x, y)
    test = dr.parseFile("data/test.txt")
    xtruth, ytruth = test[:, :-1], test.T[-1]
    predictions = tree.predict(test)
    e = Evaluator()
    a = e.confusion_matrix(ytruth, predictions)
    print("confusion" + "\n" + str(a))
    print("accuracy: " + str(e.accuracy(a)))
    print("Recall: " + str(e.recall(a)))
    print("Precision: " + str(e.precision(a)))
    print("F1score: " + str(e.f1_score(a)))

    print("RESULTS FOR TRAIN_SUB.TXT:")
    data = dr.parseFile("data/train_sub.txt")
    x, y = data[:, :-1], data.T[-1]
    tree = cp.DecisionTreeClassifier()
    tree.train(x, y)
    test = dr.parseFile("data/test.txt")
    xtruth, ytruth = test[:, :-1], test.T[-1]
    predictions = tree.predict(test)
    e = Evaluator()
    a = e.confusion_matrix(ytruth, predictions)
    print("confusion" + "\n" + str(a))
    print("accuracy: " + str(e.accuracy(a)))
    print("Recall: " + str(e.recall(a)))
    print("Precision: " + str(e.precision(a)))
    print("F1score: " + str(e.f1_score(a)))
