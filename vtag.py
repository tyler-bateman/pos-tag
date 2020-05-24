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
        totalLog = log(sum(twCounts[tag].values()))
        tw[tag] = {}
        for word in twCounts[tag]:
            tw[tag][word] = log(twCounts[tag][word]) - totalLog if twCounts[tag][word] > 0 else -float('inf')


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
            tt[t1][t2] = log(ttCounts[t1][t2]) - totalLog if ttCounts[t1][t2] > 0 else -float('inf')


def getViterbiTags(words):
    tmp = [dict.fromkeys(tagSet, '###')]
    trellis = [{}]

    bt = dict.fromkeys(tagSet, ['###'])

    #populate first column in trellis
    for tag in tagSet:
        emission = tw[tag][words[1].split('/')[0].strip()]
        transition = tt['###'][tag]
        trellis[0][tag] = emission + transition
        bt[tag] = bt['###'] + [tag]

    for line in words[2:]:
        word = line.split('/')[0].strip()
        col = {}
        newbt = {}
        prevCol = trellis[-1]
        for tag in tagSet:
            (prob, prevTag) = max([(prevCol[prevTag] + tt[prevTag][tag] + tw[tag][word] , prevTag) for prevTag in tagSet])
            col[tag] = prob
            newbt[tag] = bt[prevTag] + [tag]
        bt = newbt
        trellis.append(col)

    correct = 0
    for i in range(len(words)):
        if bt['###'][i] == words[i].split('/')[1].strip() and bt['###'][i] != '###':
            correct += 1

    return (bt['###'], 100 * correct/(len(words) - 2))


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

        (tags, accuracy) = getViterbiTags(testData)
        print('Tagging accuracy (Viterbi decoding): {}%'.format(accuracy))


        testFile.close()

    else:
        print("Invalid input")

if __name__ == '__main__':
    main()
