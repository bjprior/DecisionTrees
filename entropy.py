import numpy as np
import dataReader


def findBestNode(dataSet):
    maxInfoGain = 0
    splitAttribute = -1
    splitAttributeValue = -1
    parentEntropy = calcEntropy(dataSet[-1])

    for attribute in range(dataSet.shape[1] - 1):  # Final column is classification so not iterated over
        # Get array of attribute against classification sorted w.r.t attributeValue
        attributeData = dataSet[:, [attribute, -1]]
        attributeData = attributeData[attributeData[:, 0].argsort()]

        attributeValues = np.unique(dataSet[:, attribute])
        # If only 1 attribute value then cannot split over this attribute
        if len(attributeValues) == 1:
            continue

        for attributeValue in attributeValues[:-1]:  # Final attribute value not included as cannot split over it
            # Plus one as slicing inclusive of lower bound and exclusive of upper bound
            boundary = np.where(attributeData[:, 0] == attributeValue)[0][-1] + 1

            # Calc entropy of potential child dataSets adjusting for size of child sets
            entropyChild1 = (boundary / len(dataSet)) * calcEntropy(attributeData[:boundary, -1])
            entropyChild2 = (len(dataSet) - boundary) / len(dataSet) * calcEntropy(attributeData[boundary:, -1])

            infoGain = parentEntropy - entropyChild1 - entropyChild2
            # print(parentEntropy, entropyChild1, entropyChild2, infoGain)
            if infoGain > maxInfoGain:
                maxInfoGain = infoGain
                splitAttribute = attribute
                splitAttributeValue = attributeValue

    return splitAttribute, splitAttributeValue


def calcEntropy(dataSet):
    # Calc entropy of a data set by getting count of each value converting this to a probability
    # and then converting this to an entropy
    (unique, counts) = np.unique(dataSet, return_counts=True)
    probabilities = np.asarray(counts) / len(dataSet)
    return -np.sum(probabilities * np.log2(probabilities))


if __name__ == "__main__":
    dataSet = dataReader.parseFile("data/toy.txt")
    print(findBestNode(dataSet))
