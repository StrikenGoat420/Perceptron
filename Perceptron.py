import csv
import copy
import random
import numpy as np


def shorten_list (list, inp):

    if inp == 1:
        list2 = [v for v in list if v[-1] != "class-3"]
    elif inp == 2:
        list2 = [v for v in list if v[-1] != "class-2"]
    elif inp == 3:
        list2 = [v for v in list if v[-1] != "class-1"]

    return list2

def mulperceptron (train, test):
    epochs = 20
    #weights = np.random.rand(3,4)
    weights = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    bias = [0.1, 0.1, 0.1]
    lr = 0.2
    print("Press 0 if you do not want to do regularization")
    print("Press 1 if you want to do regularization")
    reg = int(input())
    reg_val = 0
    if reg == 1:
        print("Enter regularization value ")
        reg_val = float(input())




    for i in range(len(weights)):
        correct = 0
        wrong = 0
        new_train = changeclassname(train, i)
        outputs = []
        for data in new_train:
            if data[-1] not in outputs:
                outputs.append(data[-1])
        for epoch in range(epochs): #len of weights = 3 and we have 3 train cases as well
            for data in new_train:
                weights[i], bias[i] = multraining(bias[i], weights[i], data, lr, outputs, reg, reg_val)
                output = mulpred(data, weights[i], bias[i])
                if data[-1] == outputs[0]:
                    expected = 1
                elif data[-1] == outputs[1]:
                    expected = 2
                if output == expected:
                    correct+=1
                else:
                    wrong+=1

        total = correct + wrong
        accuracy = (correct/total)*100
        print()
        if i == 0:
            a = "training accuracy for class 1 vs rest"
        elif i == 1:
            a = "training accuracy for class 2 vs rest"
        elif i == 2:
            a = "training accuracy for class 3 vs rest"
        print(a+ " " +str(accuracy))

    print("\nupdated weights are ")
    for i in weights:
        print(i)
    print()
    correct = 0
    wrong = 0
    for data in test:
        output, correct, wrong = finalpred(data, weights, bias, correct, wrong)

    total = correct + wrong
    accuracy = (correct/total)*100
    print("accuracy is " +str(accuracy))


def finalpred (test, weights, bias, correct, wrong):
    sum = [0,0,0]
    test = list(test)
    for i in range(3):
        for j in range(len(weights[i])):
            sum[i] = bias[i]
            sum[i] += float(test[j])*weights[i][j]

    #print(np.argmax(sum) + 1)
    output = "class-" +str(np.argmax(sum) + 1)
    if test[-1] == output:
        correct += 1
    else :
        wrong += 1

    return np.argmax(sum), correct, wrong

def mulpred (inputs, weights, bias):
    sum = bias
    #len(inputs)-1 because the final element will be the output
    for i in range(len(inputs)-1):
        sum += float(weights[i])*float(inputs[i])
    output = activation_function(sum)
    return output

def multraining (bias, weights, data, lr, outputs, reg, val):
    if data[-1] == outputs[0]:
        expected = 1
    elif data[-1] == outputs[1]:
        expected = 2

    pred = mulpred(data, weights, bias)
    error = expected - pred
    bias = bias + (lr * error)
    for i in range(len(data) - 1):
        if reg == 0:
            weights[i] = float(weights[i]) + (float(lr) * float(error) * float(data[i]))
        elif reg == 1:
            weights[i] = weights[i] + (expected*float(data[i])) - (2*val*weights[i])

    return weights, bias


#this is a regular perceptron function, aka binary classifier
def perceptron(training_data, testing_data, weights, lr, bias, epochs):
    weights = np.random.rand(1,4)
    correct = 0
    wrong = 0
    print("----------------------------------------")
    print("To compare class 1 and class 2 ---- press 1")
    print("To compare class 1 and class 3 ---- press 2")
    print("To compare class 2 and class 3 ---- press 3")

    user_input = int(input())
    print()
    train = shorten_list(training_data, user_input)
    test = shorten_list(testing_data, user_input)

    outputs = []
    for data in train:
        if data[-1] not in outputs:
            outputs.append(data[-1])

    print ("initial weights are ", end = '')
    print (weights)
    print ("initial bias is ", end = '')
    print (bias)
    print()

    for epoch in range(epochs):
        for data in train:
            weights, bias = training(bias,weights,data,lr, outputs)
            output = prediction(data, weights, bias)
            if data[-1] == outputs[0]:
                expected = 1
            elif data[-1] == outputs[1]:
                expected = 2
            if output == expected:
                correct+=1
            else:
                wrong+=1

    total = correct + wrong
    accuracy = (correct/total)*100
    print("training accuracy is " +str(accuracy))


    print ("\nupdated weights are ", end = '')
    print (weights)
    print ("updated bias is ", end = '')
    print (bias)
    print()

    if user_input == 1:
        print("For class 1 and class 2 most important input parameter is " +str(np.argmax(weights[0])+1))
    elif user_input == 2:
        print("For class 1 and class 3 most important input parameter is " +str(np.argmax(weights[0])+1))
    elif user_input == 3:
        print("For class 2 and class 3 most important input parameter is " +str(np.argmax(weights[0])+1))


    correct = 0
    wrong = 0

    for data in test:
        predicted_output = prediction(data, weights, bias)
        if data[-1] == outputs[0]:
            actual_output = 1
        elif data[-1] == outputs[1]:
            actual_output = 2
        correct, wrong = accuracy_check(actual_output, predicted_output, correct, wrong)

    total = correct + wrong
    accuracy = (correct/total)*100
    print("accuracy for testing data is " +str(accuracy))

    return accuracy



def changeclassname (input, case):
    actual_data = copy.deepcopy(input)
    if case == 0:
        for data in actual_data:
            #isolating class 1
            #class 1 and 3
            if data[-1] == "class-2":
                data[-1] = "class-3"
    elif case == 1:
        for data in actual_data:
            #isolating class 2
            #class 2 and 3
            if data[-1] == "class-1":
                data[-1] = "class-3"
    elif case == 2:
        for data in actual_data:
            #isolating class 3
            #class 2 and 3
            if data[-1] == "class-1":
                data[-1] = "class-2"

    return actual_data





#standard accuracy check program
def accuracy_check (actual_data,predicted, correct, wrong):
    if actual_data == predicted:
        correct+= 1
    else :
        wrong+= 1

    return correct, wrong

def prediction(inputs, weights, bias):
    sum = bias
    #len(inputs)-1 because the final element will be the output
    for i in range(len(inputs)-1):
        #weights is a 2d array
        sum += float(weights[0][i])*float(inputs[i])
    output = activation_function(sum)
    return output


def activation_function(sum):
    #sigmoid activation function
    output = 1/(1 + np.exp(-sum))
    if output <= 0.5:
        return 1
    else :
        return 2

def training (bias, weights, data, lr, outputs = []):
    if data[-1] == outputs[0]:
        expected = 1
    elif data[-1] == outputs[1]:
        expected = 2

    pred = prediction(data, weights, bias)
    error = expected - pred
    bias = bias + (lr * error)
    for i in range(len(data) - 1):
        weights[0][i] = float(weights[0][i]) + (float(lr) * float(error) * float(data[i]))

    return weights, bias

# importing data
with open ('train.data', 'r') as f:
    train = list(csv.reader(f,delimiter=','))

with open ('test.data', 'r') as f:
    test = list(csv.reader(f,delimiter=','))


train = list(np.array(train, dtype=None))
test = list(np.array(test, dtype=None))


random.shuffle(train)

bias = 0.2
weights = [[0,0,0,0]]

learning_rate = 0.1
epochs = 20

print("For binary perceptron ----- press 1")
print("For multiclass perceptron - press 2")
choice = int(input())
if choice == 1:
    perceptron(train, test, weights, learning_rate, bias, epochs)
elif choice == 2:
    mulperceptron(train, test)
