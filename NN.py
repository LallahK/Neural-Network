from Perceptron import Perceptron
import numpy as np
from random import random
from math import sqrt

class NN(object):

    def __init__(self, meta, data, maxI):
        self.meta = meta
        self.data = data
        self.maxIterations = maxI

        H = int(len(data) / (4 * (meta[0][0] + meta[1][0]))) + 2
        I = meta[1][0] + 1
        O = meta[0][0]

        self.inputL = [Perceptron(i) for i in range(I)]
        self.hiddenL = [Perceptron(i) for i in range(I, I + H)]
        self.outputL = [Perceptron(i) for i in range(I + H, I + H + O)]
        self.perceptrons = self.inputL + self.hiddenL + self.outputL

        self.network = [
            [None for i in range(I)] + [self.fanin(I) for i in range(H - 1)] + [None for i in range(1 + O)]
        for j in range(I)] + \
        [
            [None for i in range(I + H)] + [self.fanin(H) for i in range(O)]
        for j in range(H)] + \
        [
            [None for i in range(I + H + O)]
        for j in range(O)]

        self.N = 0.05

    def fanin(self, fanIn):
        f = 1 / sqrt(fanIn)
        return random() * 2 * f - f

    def run(self):
        self.iteration = 0

        self.feedForward(self.data[0][0])

        while not self.stop():
            self.feedPatterns()
        
        self.classification()

    def feedPatterns(self):
        for d in self.data:
            self.feedForward(d[0])
            self.errorSig(d[1])
            self.backProp()

    def feedForward(self, d):
        for i, v in enumerate(d):
            self.inputL[i].setVal(v)

        for h in self.hiddenL[:-1]:
            h.activation(self.network, self.perceptrons)

        for o in self.outputL:
            o.activation(self.network, self.perceptrons)

        return [o.value for o in self.outputL]

    def errorSig(self, t):
        for i, o in enumerate(self.outputL):
            o.errorCalc(self.network, self.perceptrons, "O", t[i])

        for h in self.hiddenL[:-1]:
            h.errorCalc(self.network, self.perceptrons, "H")
    
    def backProp(self):
        l = len(self.perceptrons)
        for row in range(l):
            for col in range(l):
                w = self.network[row][col]
                if not w is None:
                    self.network[row][col] += (-self.N) * self.perceptrons[col].error * self.perceptrons[row].value

    def classification(self):
        theta = 0.4
        count = 0

        for d in self.data:
            o = self.feedForward(d[0])
            correct = True
            for k in range(len(o)):
                correct = (d[1][k] == 1 and o[k] >= 0.5 + theta) or \
                            (d[1][k] == 0 and o[k] <= 0.5 - theta)
            if correct:
                count += 1

        print("{}".format(count / len(self.data)))

    def printNetwork(self):
        for row in self.network:
            string = ""
            for r in row:
                v = "{:.3f}".format(r if not r is None else 0)
                string = "{} {:>6}".format(string, v)
            print(string)

    def stop(self):
        self.iteration += 1

        if self.iteration > self.maxIterations:
            return True

        return False