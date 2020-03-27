import csv
import copy
import random
import numpy as np

'''The first 4 coloumns are the input and the final coloumn is the desired output.
    Random weights have been assigned in the main function. Data will go one row at a time into
    the training function, this is where the weight updation will be happening based on the prediction.
    The perceptron function is the one containing the actual perceptron algorithm. The output of which
    depends on the activation function.

    We are using the sigmoid activation function, because ours is a binary classifier. The sigmoid function
    will give output in the range of 0 to 1. In our case if the output is less then 0.5, prediction will be
    1 and if the output is more than 0.5 prediction will be 2.

    To find the most discriminatory input parameter, our weight training and activation function will remain
    the same. What we will do is instead of passing all 4 input parameters, we will pass 3 input parameters.
    The lower the accuracy after removing each parameter, the more important that parameter is, aka more
    discriminatory. Weight assignment will not be random, this will lead to a more consistent answer.

    For the multiclass classifier, in the first instance we change all class 1 to class 2. Then run the perceptron
    as a binary perceptron. We then look at the probability of the data being in class 3. This thing is then repeated
    by changing class 1 into class 3, then again look at the probability of the data being in class 2. Then we change
    class 2 into class 3, then look at the probability of the data being in class 1. The one with the highest probability
    will be our final prediction

    # TODO: Improve accuracy of multiclass classifier
            Remove the need for user intervention in binary classifier (ie. user need not specify which
            two classes he/she wants to compare. by default compare all three classes and report accuracy)
            For most important parameter, return argmax of weights + 1 as well, alongside the implemented
            solution.
'''

def shorten_list (list, inp):

    if inp == 1:
        list2 = [v for v in list if v[-1] != "class-3"]
    elif inp == 2:
        list2 = [v for v in list if v[-1] != "class-2"]
    elif inp == 3:
        list2 = [v for v in list if v[-1] != "class-1"]

    return list2

def mulperceptron (train, test):
    epochs = 1000
    weights = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    bias = [0.1, 0.1,0.1]
    lr = 0.2
    for epoch in range(epochs):
        for i in range(3):
            new_train = changeclassname(train, i)
            outputs = []
            for data in new_train:
                if data[-1] not in outputs:
                    outputs.append(data[-1])
            for data in new_train:
                weights[i], bias[i] = multraining(bias[i], weights[i], data, lr, outputs)

    print("updated weights are ")
    for i in weights:
        print(i)

    correct = 0
    wrong = 0
    for data in test:
        output, correct, wrong = finalpred(data, weights, bias, correct, wrong)

    total = correct + wrong
    accuracy = (correct/total)*100
    print("correct is " +str(correct))
    print("total is "+str(total))
    print("accuracy is " +str(accuracy))



def finalpred (test, weights, bias, correct, wrong):
    sum = [0,0,0]
    test = list(test)

    for i in range(3):
        for j in range(len(weights[i])):
            sum[i] = bias[i]
            sum[i] += float(test[j])*weights[i][j]

    print(np.argmax(sum) + 1)
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
        #weights is a 2d array
        sum += float(weights[i])*float(inputs[i])
    output = activation_function(sum)
    return output

def multraining (bias, weights, data, lr, outputs):
    if data[-1] == outputs[0]:
        expected = 1
    elif data[-1] == outputs[1]:
        expected = 2

    pred = mulpred(data, weights, bias)
    error = expected - pred
    bias = bias + (lr * error)
    for i in range(len(data) - 1):
        weights[i] = float(weights[i]) + (float(lr) * float(error) * float(data[i]))
        #weights[i] = weights[i] + (expected*float(data[i])) - (2*0.1*weights[i])

    return weights, bias


#this is a regular perceptron function, aka binary classifier
def perceptron(training_data, testing_data, weights, lr, bias, epochs):
    correct = 0
    wrong = 0

    print("To compare class 1 and class 2 ---- press 1")
    print("To compare class 1 and class 3 ---- press 2")
    print("To compare class 2 and class 3 ---- press 3")

    user_input = int(input())
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

    for epoch in range(epochs):
        for data in train:
            weights, bias = training(bias,weights,data,lr, outputs)

    print ("updated weights are ", end = '')
    print (weights)
    print ("updated bias is ", end = '')
    print (bias)

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



def changeclassname (input, case):
    actual_data = copy.deepcopy(input)
    if case == 0:
        for data in actual_data:
            #isolating class 3
            #class 2 and 3
            if data[-1] == "class-1":
                data[-1] = "class-2"
    elif case == 1:
        for data in actual_data:
            #isolating class 2
            #class 2 and 3
            if data[-1] == "class-1":
                data[-1] = "class-3"
    elif case == 2:
        for data in actual_data:
            #isolating class 1
            #class 1 and 3
            if data[-1] == "class-2":
                data[-1] = "class-3"

    return actual_data


#function to create new list, by removing one input parameter at a time. Used to find
#most imporatant parameter
def create_newlist(train, test, col):
    new_train = []
    new_test = []

    print("remove col " +str(col))

    for j in train:
        j = j.tolist()
        j.pop(col)
        new_train.append(j)

    for j in test:
        j = j.tolist()
        j.pop(col)
        new_test.append(j)

    return new_train, new_test

#this function will tell us which input parameter is the most discriminatory. aka q5
def impparameters (train, test, lr, epochs):
    new_train = []
    new_test = []
    weights = [[0,0,0,0]]
    bias = 0.2

    for i in range(4):
        new_train, new_test = create_newlist(train, test, i)
        random.shuffle(new_train)
        random.shuffle(new_test)
        if i == 0:
            print("training data consists of x1,x2,x3")
        elif i == 1:
            print("training data consists of x0,x2,x3")
        elif i == 2:
            print("training data consists of x0,x1,x3")
        elif i == 3:
            print("training data consists of x0,x1,x2")
        perceptron(new_train, new_test, weights, lr, bias, epochs)
        print('----------------------------')
        weights = [[0,0,0,0]]



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
#weights = [[0,0,0,0]]
weights = np.random.rand(1,4)
learning_rate = 0.1
epochs = 20

resume = True

while resume:
    print()
    print("For Binary Perceptron ------ press 1")
    print("For Multiclass Perceptron -- press 2")
    print("To exit press anything else")
    choice = input()
    if choice == '1':
        perceptron(train, test, weights, learning_rate, bias, epochs)
    elif choice == '2':
        mulperceptron(train, test)
    else:
        resume = False
