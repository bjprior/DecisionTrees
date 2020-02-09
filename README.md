## CO395 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.

### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``eval.py``

	* Contains the skeleton code for the ``Evaluator`` class. Your task is to 
implement the ``confusion_matrix()``, ``accuracy()``, ``precision()``, 
``recall()``, and ``f1_score()`` methods.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods defined in ``classification.py`` and ``eval.py``.


### Instructions

In order to run this program, you do not need to build any code. Simply run the ``main`` method in the ``main.py`` code.


The ``main`` method will:
- load the data set from the file's path
- train the decision tree 
- Evaluate the predictions using a confusion matrix, computing accuracy, macro-precision and macro-recall
- Prune the decision tree
- Evaluate changes post-pruning

These steps will be printed to the ``terminal`` for user-readability


### Important Consideration
You may choose to change the data sets by changing the pathnames inside the ``parseFile`` method.
Were you to use a different data set from those provided in this Coursework, a few considerations on the type of data 
ought to be outlined. Each instance of the data set should:

- Should be on a separate line of the file

- Contain real-number attributes, seperated by a ','

- Contain a single classification. This will be coverted to its ASCII value in the implementation



