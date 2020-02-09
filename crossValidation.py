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
    prePruneConfMatrix = []
    postPruneConfMatrix = []

    for i in range(1, k + 1):
        testing, validation, training = split_set(data_set, k, i)
        training_x = training[:, :-1]
        training_y = [chr(i) for i in training.T[-1]]
        testing_y = [chr(i) for i in testing.T[-1]]

        tree.train(training_x, training_y)
        predictions = tree.predict(testing)
        confusion = ev.Evaluator.confusion_matrix(predictions, testing_y)
        prePruneConfMatrix.append(confusion)

        tree.prune((validation[:, :-1], [chr(i) for i in validation[:, -1]]))
        predictions = tree.predict(testing)
        confusion = ev.Evaluator.confusion_matrix(predictions, testing_y)
        accuracy[i-1] = ev.Evaluator.accuracy(confusion)
        if accuracy[i-1] > max_accuracy:
            best_tree = tree
        print("Post prune Accuracy(" + str(i) + "):" + str(accuracy[i-1]))
        postPruneConfMatrix.append(confusion)

    print("Pre pruning metrics")
    analyseListOfConfMatrix(prePruneConfMatrix)
    print("Post pruning results")
    analyseListOfConfMatrix(postPruneConfMatrix)

    return accuracy, best_tree


def analyseListOfConfMatrix(confMatrixList):
    metrics = []
    for confMatrix in confMatrixList:
        foldMetrics = [ev.Evaluator.accuracy(confMatrix), ev.Evaluator.precision(confMatrix)[1],
                       ev.Evaluator.recall(confMatrix)[1], ev.Evaluator.f1_score(confMatrix)[1]]
        metrics.append(foldMetrics)

    metrics = np.array(metrics)
    print("Accuracy: " + str(np.mean(metrics[:, 0])) + " " + str(np.std(metrics[:, 0])))
    print("Precision: " + str(np.mean(metrics[:, 1])) + " " + str(np.std(metrics[:, 1])))
    print("Recall: " + str(np.mean(metrics[:, 2])) + " " + str(np.std(metrics[:, 2])))
    print("F1: " + str(np.mean(metrics[:, 3])) + " " + str(np.std(metrics[:, 3])))


def k_decision_trees(training, testing, k):
    trees = []
    predictions = list()

    for i in range(1, k + 1):
        trees.append(cls.DecisionTreeClassifier())
        testing_new, validation, training_new = split_set(training, k, i)
        training_x = training_new[:, :-1]
        training_y = [chr(i) for i in training_new.T[-1]]
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
            if i == fold % k:
                testing_set = np.array(data_splits[i])
            elif i == (fold + 1) % k:
                validation_set = np.array(data_splits[i])
            else:
                training_set = np.concatenate((training_set, data_splits[i]), axis=0)

        training_set = np.asarray(training_set)

        return testing_set, validation_set, training_set


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
    full_data = dr.mergeAttributesAndCharacteristics(full_data[0], full_data[1])
    print(full_data)
    test_data = dr.mergeAttributesAndCharacteristics(test_data[0], test_data[1])
    k = 10
    n = len(full_data) / k
    accuracy, cross_tree = k_fold_cross_validation(full_data, k)

    # Print Accuracies and Standard Deviations for Question 3.3
    std_dev = standard_dev(accuracy, k, n)

    for i in range(len(accuracy)):
        print(str(round(accuracy[i], 4)) + " ± " + str(round(std_dev[i], 4)))

    # Question 3.4
    x = full_data[:, :-1]
    y = [chr(i) for i in full_data.T[-1]]
    Full_trained = cls.DecisionTreeClassifier()
    Full_trained.train(x, y)
    testing_y = [chr(i) for i in test_data.T[-1]]
    full_predict = Full_trained.predict(test_data)
    cross_predict = cross_tree.predict(test_data)

    print_results(full_predict, testing_y, "Fully Trained")
    print_results(cross_predict, testing_y, "K-Fold Trained")

    # Question 3.5
    k_predict = k_decision_trees(full_data, test_data, k)

    print_results(k_predict, testing_y, "K-Fold Mode Predict")

 #   plt.figure()
 #    depth = np.array([18, 18, 20, 16, 17, 20, 17, 18, 18, 17])
 #    accuracy = np.array([0.84102564, 0.87692308, 0.88461538, 0.85641026, 0.83846154, 0.85384615,
 # 0.83589744, 0.84358974, 0.83589744, 0.85897436])
 #    plt.scatter(depth, accuracy, label="train_noisy.txt unpruned")
 #    depth = np.array(([15, 15, 16, 15, 16, 16, 15, 17, 16, 15]))
 #    accuracy = np.array([0.82051282, 0.83846154, 0.87948718, 0.85641026, 0.83333333, 0.87179487,
 # 0.85641026, 0.83333333, 0.82307692, 0.87692308])
 #    plt.scatter(depth, accuracy, label="train_noisy.txt pruned")
 #    depth = np.array([19, 19, 22, 19, 19, 21, 20, 18, 20, 20])
 #    accuracy = np.array([0.90769231, 0.92051282, 0.91282051, 0.93589744, 0.90769231, 0.9025641,
 # 0.88461538, 0.91538462, 0.92307692, 0.88974359])
 #    plt.scatter(depth, accuracy, label="train_full.txt unpruned")
 #    depth = np.array([16, 17, 17, 17, 18, 19, 19, 14, 17, 19])
 #    accuracy = [0.88461538, 0.88461538, 0.87435897, 0.90512821, 0.9, 0.9,
 #     0.87692308, 0.86410256, 0.89487179, 0.87692308]
 #    plt.scatter(depth, accuracy, label="train_full.txt pruned")
 #
 #    # print(np.mean(depth))
 #    # print(np.std(depth))
 #    plt.xlabel("Max Node Depth")
 #    plt.ylabel("Accuracy (%)")
 #    plt.legend()
 #    plt.savefig("PruningAccuracy.png")
 #    plt.show()
