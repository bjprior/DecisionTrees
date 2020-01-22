import numpy as np


class Data:

    def __init__(self, attribute_array, characteristic):
        self.attribute_array = attribute_array
        self.characteristic = characteristic

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return str(self.attribute_array) + ": " + str(self.characteristic) + "\n"


def parseFile(fname):
    data = []

    with open(fname) as file:
        for line in file:
            line = line.strip()
            characteristic = line[-1]
            line = line[:-2]
            attribute_array = [int(digit) for digit in line.split(',')]
            data.append(Data(attribute_array, characteristic))

    return np.array(data)


if __name__ == "__main__":
    inputData = parseFile("data/train_full.txt")
    print(inputData)
