import numpy as np
import dataReader as dr
import classification as cls
import eval as ev
from scipy import stats


def k_fold_cross_validation(data_set, k):
    accuracy = np.zeros(k)
    tree = cls.DecisionTreeClassifier()
    best_tree = cls.DecisionTreeClassifier()
    max_accuracy = 0

    for i in range(1, k + 1):
        testing, training = split_set(data_set, k, i)
        training_x, training_y = training[:, :-1], training.T[-1]
        tree.train(training_x, training_y)
        predictions = tree.predict(testing)
        eval = ev.Evaluator()
        testing_y = testing.T[-1]
        # print("Prediction(" + str(i) + "): " + str(predictions))
        # print("Label(" + str(i) + ")" + str(testing_y))
        confusion = eval.confusion_matrix(predictions, testing_y)
        accuracy[i - 1] = eval.accuracy(confusion)
        if accuracy[i - 1] > max_accuracy:
            best_tree = tree
        # print("Accuracy(" + str(i) + "):" + str(accuracy[i - 1]))

    return accuracy, best_tree


def k_decision_trees(training, testing, k):
    trees = []
    predictions = list()

    for i in range(1, k + 1):
        trees.append(cls.DecisionTreeClassifier())
        testing_new, training_new = split_set(training, k, i)
        training_x, training_y = training_new[:, :-1], training_new.T[-1]
        trees[i - 1].train(training_x, training_y)
        predictions.append(trees[i - 1].predict(testing))

    prediction = np.array(predictions)
    prediction.astype(str)
    best_predictions = np.zeros(len(testing))
    best_predictions.astype(str)

    best_predictions = stats.mode(prediction, axis=0)[0]

    return np.array(best_predictions[0, :])


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


# Answer for
def standard_dev(accuracy, k, n):
    errors = np.ones(k) - accuracy
    std_dev = np.sqrt((errors * accuracy) / n)

    return std_dev


def print_results(predictions, labels, name):
    eval = ev.Evaluator()

    confusion = eval.confusion_matrix(predictions, labels)
    accuracy = eval.accuracy(confusion)
    precision = eval.precision(confusion)
    recall = eval.recall(confusion)
    f1_score = eval.f1_score(confusion)

    print(" ")
    print(" ")
    print("Summary evaluation for " + str(name))
    print("____________________________________")
    print("Confusion Matrix: ")
    print(str(confusion))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1_score))
    print("____________________________________")


if __name__ == "__main__":
    # Data Imports
    full_data = dr.parseFile("data/train_full.txt")
    test_data = dr.parseFile("data/test.txt")

    k = 10
    n = len(full_data) / k
    accuracy, cross_tree = k_fold_cross_validation(full_data, k)

    # Print Accuracies and Standard Deviations for Question 3.3
    std_dev = standard_dev(accuracy, k, n)

    for i in range(len(accuracy)):
        print(str(round(accuracy[i], 4)) + " ± " + str(round(std_dev[i], 4)))

    # Question 3.4
    x, y = full_data[:, :-1], full_data.T[-1]
    Full_trained = cls.DecisionTreeClassifier()
    Full_trained.train(x, y)
    testing_y = test_data.T[-1]
    full_predict = Full_trained.predict(test_data)
    cross_predict = cross_tree.predict(test_data)

    print_results(full_predict, testing_y, "Fully Trained")
    print_results(cross_predict, testing_y, "K-Fold Trained")

    # Question 3.5
    k_predict = k_decision_trees(full_data, test_data, k)

    print_results(k_predict, testing_y, "K-Fold Mode Predict")
