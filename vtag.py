from math import log, exp

#The transition probabilities in log space
tt = {}

#The emission probabilities in log space
tw = {}

tagSet = ['H', 'C', '###']
vocab = ['1', '2', '3', '###']

def trainEmission(trainData):
    twCounts = dict.fromkeys(tagSet, {})
    for tag in twCounts:
        twCounts[tag] = dict.fromkeys(vocab, 0)

    for line in trainData:
        (word, tag) = line.split('/')
        word = word.strip()
        tag = tag.strip()
        twCounts[tag][word] += 1

    for tag in twCounts:
        #print(tag)
        totalLog = log(sum(twCounts[tag].values()))
        tw[tag] = {}
        for word in twCounts[tag]:

            tw[tag][word] = log(twCounts[tag][word]) - totalLog if twCounts[tag][word] > 0 else None

def trainTransition(trainData):
    ttCounts = dict.fromkeys(tagSet, {})
    for tag in ttCounts:
        ttCounts[tag] = dict.fromkeys(tagSet, 0)


    for i in range(len(trainData) - 1):
        t1 = trainData[i].split('/')[1].strip()
        t2 = trainData[i + 1].split('/')[1].strip()
        if t1 in ttCounts:
            if t2 in ttCounts[t1]:
                ttCounts[t1][t2] += 1
            else:
                ttCounts[t1][t2] = 1
        else:
            ttCounts[t1] = {t2:1}

    for t1 in ttCounts:
        totalLog = log(sum(ttCounts[t1].values()))
        tt[t1] = {}
        for t2 in ttCounts[t1]:
            tt[t1][t2] = log(ttCounts[t1][t2]) - totalLog if ttCounts[t1][t2] > 0 else None


def getViterbiTags(words):
    #Initialize trellis with None
    vTrellis = [dict.fromkeys(tagSet, None) for word in words]
    vTrellis[0]['###'] = 0
    bt = []
    for i in range(len(words) - 1):
        word = words[i].split('/')[0]
        #print(word)
        tagProbs = {}
        bt.append({})
        for tag in tagSet:
            maxTag = '###'
            maxProb = None
            for prevTag in tagSet:
                prevProb = vTrellis[i][prevTag]
                trans = tt[prevTag][tag] if prevTag in tt and tag in tt[prevTag] else 0
                emit = tw[tag][word] if tag in tw and word in tw[tag] else 0
                totProb = prevProb + trans + emit if prevProb != None and trans != None and emit != None else None
                if totProb != None and totProb > maxProb:
                    maxProb = totProb
                    maxTag = prevTag
            bt[i][tag] = maxTag
    print(bt)
    tagVector = ['###']
    curTag = bt[-1]['###']
    i = -1
    while curTag != '###':
        print(curTag)
        tagVector.insert(0, curTag)
        curTag = bt[i][curTag]
        i -= 1

    return tagVector


def main():
    from sys import argv
    if len(argv) == 3:
        trainFile = open('data/ic/{}.txt'.format(argv[1]), 'r')
        trainData = [line for line in trainFile]
        trainEmission(trainData)
        trainTransition(trainData)
        trainFile.close()

        testFile = open('data/ic/{}.txt'.format(argv[2]), 'r')
        testData = [line for line in testFile]

        print('\nViterbi Tags')
        for tag in getViterbiTags(testData):
            print(tag)
        testFile.close()

    else:
        print("Invalid input")

if __name__ == '__main__':
    main()
