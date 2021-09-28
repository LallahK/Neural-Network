from Data import readData
from time import time
from NN import NN

datasets = [
    "balance-scale",
    "bupa",
    "haberman",
    "ionosphere",
    "iris",
    "wine"
]

def readInput(inp):
    line = inp.split(' ')
    if len(line) != 2:
        return "FALSE"

    try:
        maxI = int(line[1])
        datasets.index(line[0])
    except:
        return "FALSE"

    return [line[0], maxI]

def main():
    print("-- Simple Stochastic Neural Network --")
    print("Datasets:")
    for d in datasets:
        print("{}{}".format(' ' * 4, d))
    while True:
        print("\nInput: <Dataset> <Max Iterations>")
        inp = input(">> ")
        out = readInput(inp)
        if out == "FALSE":
            print("Invalid input")
        else:
            data, meta = readData("Data/{}".format(out[0]))

            nn = NN(meta, data, out[1])
            nn.run()

if __name__ == "__main__":
    main()