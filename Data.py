import numpy as np
from random import shuffle

def readData(file):
    f = open(file + ".data")
    count, ind, meta = readInfo(file)

    data = []
    counts = [0 for i in range(meta[0][0])]

    for line in f.readlines():
        content = line.strip().split(',')
        x = [0 for i in range(meta[0][0])]
        x[meta[0][1].index(content[ind])] = 1

        y = []
        for i in range(count):
            if not i == ind:
                y.append(float(content[i]))

        counts[meta[0][1].index(content[ind])] += 1

        data.append([np.array(y), np.array(x)])

    shuffle(data)
    meta[0].append(counts)

    normalize(data, meta)

    return data, meta

def readInfo(file):
    f = open(file + '.info')

    line = f.readline().strip().replace(' ', '').split(',')
    line = [int(x) for x in line]
    index = line[1]
    count = int(line[0])

    line = f.readline().strip().replace(' ', '').split(',')
    targ = [line[i] for i in range(1, len(line))]

    fs = [line.strip() for line in f.readlines()]

    fmeta = [[len(targ), targ], [len(fs), fs]]

    return count, index, fmeta

def normalize(data, meta):
    vecs = [[] for i in range(meta[1][0])]

    for d in data:
        for i, v in enumerate(d[0]):
            vecs[i].append(v)

    vectors = [np.array(v) for v in vecs]
    means = [np.mean(v) for v in vectors]
    devs = [1 / np.std(v) for v in vectors]

    for i in range(len(data)):
        d = np.multiply(np.subtract(data[i][0], means), devs)
        data[i] = [d, data[i][1]]