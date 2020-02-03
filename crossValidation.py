import numpy as np
import dataReader as dr
import classification as cls
import eval as ev
import math


def k_fold_cross_validation(data_set, k):
    errors = np.zeros(k)
    tree = cls.DecisionTreeClassifier()

    for i in range(1, k):
        testing, training = split_set(data_set, k, i)
        training_x, training_y = training[:, :-1], training.T[-1]
        tree.train(training_x, training_y)
        predictions = tree.predict(testing)
        eval = ev.Evaluator()
        confusion = eval.confusion_matrix(predictions, training_y)
        errors[i - 1] = eval.accuracy(confusion)

    return np.mean(errors)


def split_set(data_set, k, fold):
    if fold > k or fold < 1:
        print("Incorrect usage: fold value greater than k")
        return
    elif k > len(data_set) or k < 2:  # Check for error in k input and return error if so
        print("Incorrect usage: Split value, k greater than sample size or less than 2")
        return
    else:
        width = len(data_set[0])
        data_splits = np.split(data_set,k)
        #np.random.shuffle(data_set) <- randomises the set
        training_set = np.empty(shape=[0, width], dtype=int)

        for i in range(len(data_splits)):
            if i == fold - 1:
                testing_set = np.array(data_splits[i])
            else:
                training_set = np.concatenate((training_set,data_splits[i]),axis=0)

        training_set = np.asarray(training_set)

        return testing_set, training_set


if __name__ == "__main__":
    data = dr.parseFile("data/toy.txt")
    n = len(data)
    k = 10
    average_accuracy = k_fold_cross_validation(data, k)
    #err = 1 - average_accuracy
    #standard_deviation = math.sqrt((err*(1-err))/n)