import numpy as np
import matplotlib.pyplot as plt
import string as stg

# places data from file in a numpy array
# note: characteristics in final column are treated as their ascii value
def parseFile(fname):
    data = []

    with open(fname) as file:
        for line in file:
            line = line.strip()
            characteristic = line[-1]
            line = line[:-2]
            attribute_array = [int(digit) for digit in line.split(',')]
            attribute_array.append(ord(characteristic))
            data.append(attribute_array)

    return np.array(data)

# counts the occurences of a certain character (i.e. letter) in the dataset
def characteristicOccurence(data):
    characteristicCount = np.zeros([26], dtype=int)

    characteristicData = data[:, -1]
    for letter in stg.ascii_uppercase:
        characteristicCount[ord(letter) - 65] = np.count_nonzero(characteristicData == ord(letter))

    return np.array(characteristicCount)

# create a subarray that only considers characteristics with value
def shrunkCharacteristicDataset(data):
    shrunked = []

    countCharacteristics = characteristicOccurence(data)
    index = 0
    for char in countCharacteristics:
        if(char > 0):
            shrunked.append([index + 65, char])
        index += 1
    return np.array(shrunked)

# helper function to convert a numpy array of characteristics from ascii to their corresponding alpha value.
def asciiArrayConverter(data):
    charVersionArray = []
    index = 0
    for char in data:
        charVersionArray.append(chr(char))
        index += 1
    return np.array(charVersionArray)

# returns the average of each attribute in an array as well as the min/max of each attribute in other array
def averageMaxMinAttributes(data):
    av = []
    mM = []
    dataTranspose = data.T

    for i in dataTranspose:
        av.append([np.average(i)])
        mM.append([min(i), max(i)])
    return np.array(av), np.array(mM)

# counts the number of inconsistent characteristic attributions between truth and noisy
def numberInstanceNoisy(truth, noisy):
    counter = 0
    j = 0
    for i in truth[: , -1]:
        if (i != noisy[j][-1]):
            counter += 1
        j += 1
    return counter


if __name__ == "__main__":
    train_fullData = parseFile("data/train_full.txt")
    train_noisyData = parseFile("data/train_noisy.txt")
    train_subData = parseFile("data/train_sub.txt")


    # Comparing the 3 Datasets

    ## Bar Chart: size of each Dataset (number of columns)
    train_fullSize = np.size(train_fullData[:, -1])
    train_noisySize = np.size(train_noisyData[:, -1])
    train_subSize = np.size(train_subData[:, -1])
    height = [train_fullSize, train_noisySize, train_subSize]
    x_axis = [1, 2, 3]
    bars = ('train_full', 'train_noisy', 'train_sub')
    plt.bar(x_axis, height, tick_label = bars)
    plt.ylabel('Size')
    plt.xlabel('Dataset')
    plt.title('Size of Datasets')
    plt.show()

    ## Bar Chart: number of occurences of characteristics for all Datasets
    train_fullLabelOcc = characteristicOccurence(train_fullData)
    print(train_fullLabelOcc)
    train_noisyLabelOcc = characteristicOccurence(train_noisyData)
    train_subLabelOcc = characteristicOccurence(train_subData)
    n_groups = 26
    index = np.arange(n_groups)
    bars = ('train_full', 'train_noisy', 'train_sub')
    bar_width = 0.2
    rects1 = plt.bar(index - bar_width, train_fullLabelOcc, width = 0.2,  color = 'b', label='train_full')
    rects2 = plt.bar(index, train_noisyLabelOcc, width = 0.2, color = 'r', label='train_noisy')
    rects3 = plt.bar(index + bar_width, train_subLabelOcc, width = 0.2, color = 'g', label='train_sub')
    plt.xticks(index, ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'))
    plt.ylabel('Size Occurence')
    plt.xlabel('Characteristic')
    plt.title('Characteristic Occurences for each Dataset')
    plt.legend()
    plt.show()

    # train_full Dataset

    ## Pie Chart: proportion occurence of each characteristic
    shrunkedfull = shrunkCharacteristicDataset(train_fullData)
    fig, ax = plt.subplots()
    countLetters = shrunkedfull[:, 1]
    corespondingLetters = asciiArrayConverter(shrunkedfull[:, 0])
    labels = corespondingLetters
    sizes = countLetters
    # Analysis of the data
    mu = np.average(shrunkedfull[:, 1])
    sigma = np.std(shrunkedfull[:, -1])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Characteristic Occurence for train_full')
    plt.show()

    # train_noisy Dataset

    ## Pie Chart: proportion occurence of each characteristic
    shrunkedfull = shrunkCharacteristicDataset(train_noisyData)
    fig, ax = plt.subplots()
    countLetters = shrunkedfull[:, 1]
    corespondingLetters = asciiArrayConverter(shrunkedfull[:, 0])
    labels = corespondingLetters
    sizes = countLetters
    # Analysis of the data
    mu = np.average(shrunkedfull[:, 1])
    sigma = np.std(shrunkedfull[:, -1])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Characteristic Occurence for train_noisy')
    plt.show()

    # train_sub Dataset
    ## Pie Chart: proportion occurence of each characteristic
    shrunkedfull = shrunkCharacteristicDataset(train_subData)
    fig, ax = plt.subplots()
    countLetters = shrunkedfull[:, 1]
    corespondingLetters = asciiArrayConverter(shrunkedfull[:, 0])
    labels = corespondingLetters
    sizes = countLetters
    # Analysis of the data
    mu = np.average(shrunkedfull[:, 1])
    sigma = np.std(shrunkedfull[:, -1])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Characteristic Occurence for train_sub')
    plt.show()

    # ######### Attribute Analysis ###################

    attributes = train_fullData[:, 0:-1]
    fullAverageAttribute, fullMinMaxAttributes = averageMaxMinAttributes(train_fullData[:, 0:-1])
    noisyAverageAttribute, noisyMinMaxAttributes = averageMaxMinAttributes(train_noisyData[:, 0:-1])
    subAverageAttribute, subMinMaxAttributes = averageMaxMinAttributes(train_subData[:, 0:-1])
    #print(minMaxAttributes)
    # n_groups = np.size(train_fullData[:1]) - 1
    # print(n_groups)
    # index = np.arange(n_groups)
    # bars = ('train_full', 'train_noisy', 'train_sub')
    # bar_width = 0.2
    # rects1 = plt.bar(index - bar_width, fullAverageAttribute, width = 0.2,  color = 'b', label='train_full')
    # rects2 = plt.bar(index, noisyAverageAttribute, width = 0.2, color = 'r', label='train_noisy')
    # rects3 = plt.bar(index + bar_width, subAverageAttribute, width = 0.2, color = 'g', label='train_sub')
    # plt.xticks(index, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'))
    # plt.ylabel('Size Occurence')
    # plt.xlabel('Characteristic')
    # plt.title('Characteristic Occurences for each Dataset')
    # plt.legend()
    # plt.show()

    ## QUESTION 1.3

    print(numberInstanceNoisy(train_fullData, train_noisyData)/np.size(train_fullData[:, -1]))



