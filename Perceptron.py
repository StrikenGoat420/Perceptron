import csv
import numpy as np

def activation_function(sum):
    #temp
    if (sum > 0 ):
        return sum
    elif return 0

def perceptron(inputs, weights, bias):
    sum = bias
    for i in inputs:
        sum += weights[i]*inputs[i]
    output = activation_function(sum)
    return output

def training(inputs, target, lr):
    # setting initial weights randomly
    weights = np.random.rand(1,4)
    output = perceptron(inputs, weights, -1)
    error = target - output
    for w in weights:
        w += error*inputs[i]*lr


# importing data
with open ('train.data', 'r') as f:
    train = list(csv.reader(f,delimiter=','))

with open ('test.data', 'r') as f:
    test = list(csv.reader(f,delimiter=','))

train = np.array(train, dtype=None)
test = np.array(test, dtype=None)
epochs = 20

for i in range(epochs):
    train(inputs, target, 0.1)


lr = 3
