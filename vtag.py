from math import log, exp

#The transition probabilities in log space
tt = {}

#The emission probabilities in log space
tw = {}


def trainEmission(trainData):
    twCounts = {}
    for line in trainData:
        with line.split('/') as (word, tag):
            if tag in twCounts:
                if word in twCounts[tag]:
                    twCounts[tag][word] += 1
                else:
                    twCounts[tag][word] = 1
            else:
                twCounts[tag] = {word:1}

    for tag in twCounts:
        with log(sum(tag.values())) as totalLog:
            for word in tag:
                tw[tag][word] = log(tag[word]) - totalLog

def trainTransition(trainData):
    ttCounts = {}
    for i in range(len(trainData) - 1):
        with trainData[i].split('/')[1] as t1, trainData[i + 1].split('/')[1] as t2:
            if t1 in ttCounts:
                if t2 in ttCounts[t1]:
                    ttCounts[t1][t2] += 1
                else:
                    ttCounts[t1][t2] = 1
            else:
                ttCounts[t1] = {t2:1}

    for t1 in ttCounts:
        with log(sum(tag.values())) as totalLog:
            for t2 in t1:
                tt[t1][t2] = log(t1[t2]) - totalLog
