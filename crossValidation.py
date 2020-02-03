import numpy as np
import dataReader as dr
import classification as cls
import eval as ev

def k_fold_cross_validation(data_set, k):

    accuracy = np.zeros(k)
    tree = cls.DecisionTreeClassifier()
    best_tree = cls.DecisionTreeClassifier()
    max_accuracy = 0

    for i in range(1, k+1):
        testing, training = split_set(data_set, k, i)
        training_x, training_y = training[:, :-1], training.T[-1]
        tree.train(training_x, training_y)
        predictions = tree.predict(testing)
        eval = ev.Evaluator()
        testing_y = testing.T[-1]
        print("Prediction(" + str(i) + "): " + str(predictions))
        print("Label(" + str(i) + ")" + str(testing_y))
        confusion = eval.confusion_matrix(predictions, testing_y)
        accuracy[i - 1] = eval.accuracy(confusion)
        if accuracy[i-1] > max_accuracy:
            best_tree = tree
        print("Accuracy(" + str(i) + "):" + str(accuracy[i-1]))

    return accuracy, best_tree


def split_set(data_set, k, fold):
    if fold > k or fold < 1:
        print("Incorrect usage: fold value greater than k")
        return
    elif k > len(data_set) or k < 2:  # Check for error in k input and return error if so
        print("Incorrect usage: Split value, k greater than sample size or less than 2")
        return
    else:
        width = len(data_set[0])
        data_splits = np.split(data_set, k)
        training_set = np.empty(shape=[0, width], dtype=int)

        for i in range(len(data_splits)):
            if i == fold - 1:
                testing_set = np.array(data_splits[i])
            else:
                training_set = np.concatenate((training_set, data_splits[i]), axis=0)

        training_set = np.asarray(training_set)

        return testing_set, training_set

def standard_dev(accuracy,k,n):

    errors = np.ones(k) - accuracy
    std_dev = np.sqrt((errors*accuracy)/n)

    print(std_dev)


if __name__ == "__main__":
    data = dr.parseFile("data/train_sub.txt")
    k = 10
    n = len(data)/k
    accuracy, tree = k_fold_cross_validation(data, k)
    #accuracy = [0.3333333, 0.533333, 0.7166666666, 0.383333333333, 0.4166666666, 0.48333333333, 0.48333333333
    #           , 0.36666666666666664, 0.5333333333333333, 0.43333333333333335]
    standard_dev(accuracy, k, n)

