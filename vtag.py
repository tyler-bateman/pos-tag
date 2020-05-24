from math import log, exp

#The transition probabilities in log space
tt = {}

#The emission probabilities in log space
tw = {}

tagDict = {}
tagSet = []

def buildDict(trainData):
    global tagDict, tagSet
    tagDict = {}
    tagSet = []
    for line in trainData:
        (word, tag) = line.split('/')
        tag = tag.strip()
        if not tag in tagSet:
            tagSet.append(tag)

        if word in tagDict:
            if not tag in tagDict[word]:
                tagDict[word].append(tag)
        else:
            tagDict[word] = [tag]

def trainEmission(trainData):
    twCounts = dict.fromkeys(tagSet, {})
    for tag in twCounts:
        twCounts[tag] = dict.fromkeys(tagDict.keys(), 1)

    for line in trainData:
        (word, tag) = line.split('/')
        word = word.strip()
        tag = tag.strip()
        twCounts[tag][word] += 1

    for tag in twCounts:
        totalLog = log(sum(twCounts[tag].values()))
        tw[tag] = {}
        for word in twCounts[tag]:
            tw[tag][word] = log(twCounts[tag][word]) - totalLog if twCounts[tag][word] > 0 else -float('inf')


def trainTransition(trainData):
    ttCounts = dict.fromkeys(tagSet, {})
    for tag in ttCounts:
        ttCounts[tag] = dict.fromkeys(tagSet, 1)

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
            tt[t1][t2] = log(ttCounts[t1][t2]) - totalLog if ttCounts[t1][t2] > 0 else -float('inf')


def getViterbiTags(words):
    trellis = [{}]

    bt = dict.fromkeys(tagSet, ['###'])

    #populate first column in trellis
    for tag in tagDict['###']:
        emission = tw[tag][words[1].split('/')[0].strip()]
        transition = tt['###'][tag]
        trellis[0][tag] = emission + transition
        bt[tag] = bt['###'] + [tag]

    for line in words[2:]:
        word = line.split('/')[0].strip()
        col = {}
        newbt = {}
        prevCol = trellis[-1]

        if word in tagDict:
            for tag in tagDict[word]:
                emission = tw[tag][word] if word != '###' else 0 if tag == '###' else -float('inf')
                (prob, prevTag) = max([(prevCol[prevTag] + tt[prevTag][tag] + tw[tag][word] , prevTag) for prevTag in prevCol])
                col[tag] = prob
                newbt[tag] = bt[prevTag] + [tag]
        else:
            for tag in tagSet:
                (prob, prevTag) = max([(prevCol[prevTag] + tt[prevTag][tag], prevTag) for prevTag in prevCol])
                col[tag] = prob
                newbt[tag] = bt[prevTag] + [tag]
        bt = newbt
        trellis.append(col)

    knownCount = 0
    knownCorrect = 0
    novelCount = 0
    novelCorrect = 0

    for i in range(len(words)):
        (word, cTag) = words[i].split('/')
        cTag = cTag.strip()
        gTag = bt['###'][i]
        novel = not word in tagDict
        if novel:
            novelCount += 1
        else:
            knownCount += 1
        if gTag == cTag and gTag != '###':
            if novel:
                novelCorrect += 1
            else:
                knownCorrect += 1

    return (bt['###'], 100 * (knownCorrect + novelCorrect) /(knownCount + novelCount), 100 * knownCorrect/knownCount, 100 * novelCorrect / novelCount)


def main():
    from sys import argv
    if len(argv) == 3:
        trainFile = open('data/en/{}.txt'.format(argv[1]), 'r')
        trainData = [line.strip() for line in trainFile]
        buildDict(trainData)
        trainEmission(trainData)
        trainTransition(trainData)
        trainFile.close()

        testFile = open('data/en/{}.txt'.format(argv[2]), 'r')
        testData = [line for line in testFile]

        (tags, accuracy, knownAccuracy, novelAccuracy) = getViterbiTags(testData)
        print('Tagging accuracy (Viterbi decoding): {}% (known: {}% novel: {}%)'.format(accuracy, knownAccuracy, novelAccuracy))


        testFile.close()

    else:
        print("Invalid input")

if __name__ == '__main__':
    main()
