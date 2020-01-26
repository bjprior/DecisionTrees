import numpy as np
import matplotlib.pyplot as plt
import string as stg

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

def characteristicOccurence(data):
    characteristicCount = np.zeros([26], dtype=int)

    characteristicData = data[:, -1]
    for letter in stg.ascii_uppercase:
        characteristicCount[ord(letter) - 65] = np.count_nonzero(characteristicData == ord(letter))

    return np.array(characteristicCount)

def shrunkCharacteristicDataset(data):
    shrunked = []

    countCharacteristics = characteristicOccurence(data)
    index = 0
    for char in countCharacteristics:
        if(char > 0):
            shrunked.append([index + 65, char])
        index += 1
    return np.array(shrunked)

def asciiArrayConverter(data):
    charVersionArray = []
    index = 0
    for char in data:
        charVersionArray.append(chr(char))
        index += 1
    return np.array(charVersionArray)

def averageMaxMinAttributes(data):
    av = []
    mM = []
    dataTranspose = data.T

    for i in dataTranspose:
        av.append([np.average(i)])
        mM.append([min(i), max(i)])
    return av, mM

if __name__ == "__main__":
    train_fullData = parseFile("data/train_full.txt")
    train_noisyData = parseFile("data/train_noisy.txt")
    train_subData = parseFile("data/train_sub.txt")


    ######### Comparison between the three data sets ########
    ## Size of each of the data sets
    train_fullSize = np.size(train_fullData)
    train_noisySize = np.size(train_noisyData)
    train_subSize = np.size(train_subData)
    height = [train_fullSize, train_noisySize, train_subSize]
    x_axis = [1, 2, 3]
    bars = ('train_full', 'train_noisy', 'train_sub')
    plt.bar(x_axis, height, tick_label = bars)
    plt.ylabel('Size')
    plt.xlabel('Dataset')
    plt.title('Size of Datasets')
    plt.show()


    #Size of each characteristic in the data sets
    # data to plot
    train_fullLabelOcc = characteristicOccurence(train_fullData)
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

    # ############# train_full ################

    shrunkedfull = shrunkCharacteristicDataset(train_fullData)

    ## pie chart letter occurence
    fig, ax = plt.subplots()
    countLetters = shrunkedfull[:, 1]
    corespondingLetters = asciiArrayConverter(shrunkedfull[:, 0])
    # # Data to plot
    labels = corespondingLetters
    sizes = countLetters

    # Analysis of the data
    mu = np.average(shrunkedfull[:, 1])
    sigma = np.std(shrunkedfull[:, -1])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    # Plot
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Characteristic Occurence for train_full')
    plt.show()

    # ############## train_noisy ################

    shrunkedfull = shrunkCharacteristicDataset(train_noisyData)


    ## pie chart letter occurence
    fig, ax = plt.subplots()
    countLetters = shrunkedfull[:, 1]
    corespondingLetters = asciiArrayConverter(shrunkedfull[:, 0])
    # # Data to plot
    labels = corespondingLetters
    sizes = countLetters

    # Analysis of the data
    mu = np.average(shrunkedfull[:, 1])
    sigma = np.std(shrunkedfull[:, -1])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    # Plot
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Characteristic Occurence for train_noisy')
    plt.show()

    # ############# train_sub ################

    shrunkedfull = shrunkCharacteristicDataset(train_subData)


    ## pie chart letter occurence
    fig, ax = plt.subplots()
    countLetters = shrunkedfull[:, 1]
    corespondingLetters = asciiArrayConverter(shrunkedfull[:, 0])
    # # Data to plot
    labels = corespondingLetters
    sizes = countLetters

    # Analysis of the data
    mu = np.average(shrunkedfull[:, 1])
    sigma = np.std(shrunkedfull[:, -1])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    # Plot
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Characteristic Occurence for train_sub')
    plt.show()

    # ######### Attribute Analysis ###################

    # ############## train_noisy ################

    attributes = train_fullData[:, 0:-1]
    averageAttribute, minMaxAttributes = averageMaxMinAttributes(attributes)
    print(averageAttribute)
    print(minMaxAttributes)



