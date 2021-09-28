from math import exp, sqrt
import numpy as np

class Perceptron(object):

    def __init__(self, index):
        self.index = index

        self.value = -1
        self.error = 0

    def setVal(self, val):
        self.value = val

    def activation(self, network, perceptrons):
        net = 0
        for i in range(len(perceptrons)):
            w = network[i][self.index]
            if not w is None:
                net += w * perceptrons[i].value
            
        self.value = 1 / (1 + exp(-1 * net))

    def errorCalc(self, network, perceptrons, t, target = None):
        deriv = (1 - self.value) * self.value
        if t == "O":
            self.error = (self.value - target) * deriv
        if t == "H":
            # print("-" * 50)
            sum = 0
            for i in range(len(perceptrons)):
                w = network[self.index][i]
                if not w is None:
                    # print(w)
                    sum += w * perceptrons[i].error * deriv
            self.error = sum