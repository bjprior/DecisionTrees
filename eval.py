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
    """ Class to perform evaluation
    """
    
    def confusion_matrix(self, prediction, annotation, class_labels=None):
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
        
        
        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################
                                                                                        #### NEEDS CHECKING FROM SOMEONE ELSE
        # iterate through rows and columns of the confusion matrix
        row = 0
        col = 0
        # storing count of when true letter is equal to some predicted letter
        for trueLetter in class_labels:
            for predictedLetter in class_labels:
                counter = 0
                for index in range(np.size(prediction)):
                    if (trueLetter == annotation[index] and predictedLetter == prediction[index]):
                            counter += 1
                    confusion[row][col] = counter
                row += 1
                row %= len(class_labels)
            col += 1
            col %= len(class_labels)

        return confusion
    
    
    def accuracy(self, confusion):
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
        
        # feel free to remove this
        accuracy = 0.0
        
        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        # accuracy is given by instanceWhen(TRUTH == PREDICTED) / ALL EVENTS
                                                                                        #### NEEDS CHECKING FROM SOMEONE ELSE
        truePostive = np.trace(confusion)
        allEvents = np.sum(confusion)

        accuracy = truePostive / allEvents

        return accuracy
        
    
    def precision(self, confusion):
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
        
        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################
                                                                                            #### NEEDS CHECKING FROM SOMEONE ELSE
        # precision (per characteristic) == TRUTH / TOTAL PREDICTION THAT LETTER
        index  = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            if (np.sum(confusion[:,letterIndex]) == 0):
                p[index] = 0
            else:
                p[index] = confusion[letterIndex][letterIndex] / np.sum(confusion[:,letterIndex])
            index += 1

        # You will also need to change this        
        macro_p = 0
        # finding average of the precision score for global
        macro_p = np.average(p)

        return (p, macro_p)

    
    def recall(self, confusion):
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
        r = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################
                                                                                        #### NEEDS CHECKING FROM SOMEONE ELSE
        # recall (per characteristic) == TRUTH / TOTAL TIMES THAT WAS THE TRUE LETTER
        index = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            if (np.sum(confusion[letterIndex]) == 0):
                r[index] = 0
            else:
                r[index] = confusion[letterIndex][letterIndex] / np.sum(confusion[letterIndex])
            index += 1

        # You will also need to change this        
        macro_r = 0
        # finding average of the recall score for global
        macro_r = np.average(r)

        return (r, macro_r)
    
    
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
        f = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################
                                                                                            #### NEEDS CHECKING FROM SOMEONE ELSE
        precision, macro_p = self.precision(confusion)
        recall, macro_r = self.recall(confusion)

        index = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            if (precision[index] == recall[index]):
                f[index] == 0
            else:
                f[index] = 2 * (precision[index] * recall[index]) / (recall[index] + precision[index])
            index += 1

        # You will also need to change this
        macro_f = 0

        # finding average of the f1 for global
        macro_f = np.average(f)

        return (f, macro_f)
   


if __name__ == "__main__":
    truth = np.array(["A", "B", "C", "A", "B"])
    predictions = np.array(["A", "C", "B", "D", "B"])

    e = Evaluator()
    a = e.confusion_matrix(truth, predictions)

    print(e.f1_score(a))
