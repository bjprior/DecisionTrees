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
    trees = []

    for i in range(1, k + 1):
        # Split Data into training and testing data
        testing, training = split_set(data_set, k, i)
        training_x = training[:, :-1]
        training_y = [chr(i) for i in training.T[-1]]

        # Train tree
        tree.train(training_x, training_y)
        predictions = tree.predict(testing)

        # Save Tree
        trees.append(tree)

        # Evaluation metrics
        eval = ev.Evaluator()
        testing_y = [chr(i) for i in testing.T[-1]]
        confusion = eval.confusion_matrix(predictions, testing_y)
        accuracy[i - 1] = eval.accuracy(confusion)

        # Save tree with best accuracy
        if accuracy[i - 1] > max_accuracy:
            best_tree = tree

    return accuracy, best_tree, trees


def k_decision_trees(testing, k, k_trees):
    predictions = list()

    # Get predictions for each tree
    for i in range(1, k + 1):
        predictions.append(k_trees[i - 1].predict(testing))

    prediction = np.array(predictions)
    prediction.astype(str)
    best_predictions = np.zeros(len(testing))
    best_predictions.astype(str)

    # Calculate mode for each label
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
    full_data = dr.mergeAttributesAndCharacteristics(full_data[0], full_data[1])
    test_data = dr.mergeAttributesAndCharacteristics(test_data[0], test_data[1])

    k = 10
    n = len(full_data) / k

    # Random shuffle data once
    np.random.shuffle(full_data)

    # Perform cross validation return accuracy for each tree, the best tree and an array of the k trees
    accuracy, best_tree, k_trees = k_fold_cross_validation(full_data, k)

    # print Accuracies and Standard Deviations for Question 3.3

    print("Mean: " + str(accuracy.mean()))
    print("Standard deviation: " + str(accuracy.std()))

    # Question 3.4
    x = full_data[:, :-1]
    y = [chr(i) for i in full_data.T[-1]]
    Full_trained = cls.DecisionTreeClassifier()
    Full_trained.train(x, y)
    testing_y = [chr(i) for i in test_data.T[-1]]
    full_predict = Full_trained.predict(test_data)
    cross_predict = best_tree.predict(test_data)

    print_results(full_predict, testing_y, "Fully Trained")
    print_results(cross_predict, testing_y, "K-Fold Trained")

    # Question 3.5
    k_predict = k_decision_trees(test_data, k, k_trees)

    print_results(k_predict, testing_y, "K-Fold Mode Predict")
