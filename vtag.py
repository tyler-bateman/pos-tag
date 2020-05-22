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
    bt = [dict.fromkeys(tagSet, '###')]
    trellis = [{}]

    #populate first column in trellis
    for tag in tagSet:
        emission = tw[tag][words[1].split('/')[0].strip()]
        transition = tt['###'][tag]
        trellis[0][tag] = emission + transition if emission != None and transition!= None else None

    for word in words[2:]:
        bt.append({})
        col = {}
        prevCol = trellis[-1]
        for tag in tagSet:
            maxTag = prevCol['###']
            maxVal = -float("inf")
            for tPrev in prevCol:
                if prevCol[tPrev] != None and tt[tPrev][tag] != None and prevCol[tPrev] + tt[tPrev][tag] > maxVal:
                    maxTag = tPrev
                    maxVal = prevCol[tPrev] + tt[tPrev][tag]
            bt[-1][tag] = maxTag


            emission = tw[tag][word.split('/')[0].strip()]
            col[tag] = emission + maxVal if emission != None and maxVal != None else None
        trellis.append(col)

    print(len(bt))
    print(len(words))

    tagSeq = ['###']
    correctCount = 0
    for i in range(len(words) - 1, 0, -1):
        col = bt[i - 1]
        tag = col[tagSeq[-1]]
        if tag == words[i].split('/')[1].strip():
            correctCount += 1
        tagSeq.append(tag)

    return (tagSeq, correctCount/len(words))








def main():
    from sys import argv
    if len(argv) == 3:
        trainFile = open('data/ic/{}.txt'.format(argv[1]), 'r')
        trainData = [line for line in trainFile]
        trainEmission(trainData)
        trainTransition(trainData)
        trainFile.close()
        #print(tt)
        #print(tw)

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
