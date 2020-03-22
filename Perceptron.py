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
    for o in outputs:
        print(o)

    print ("initial weights are ", end = '')
    print (weights)
    print ("initial bias is ", end = '')
    print (bias)

    for epoch in range(epochs):
        for data in train:
            weights, bias = training(bias,weights,data,lr,epochs, 0, outputs)

    print ("updated weights are ", end = '')
    print (weights)
    print ("updated bias is ", end = '')
    print (bias)

    for data in test:
        predicted_output = prediction(data, weights, bias, 0)
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

def multiclasstraining(weights, bias, input, outputs, lr):
    sum = 0
    for i in range(len(outputs)):
        if input[-1] == outputs[i]:
            actual_output = i

    pred = multiclassprediction(data, weights, bias)
    error = actual_output - pred
    bias = bias + (lr * error)
    for i in range(len(input[-1])):
        weights[i] = float(weights[i]) + (float(lr) * float(error) * float(data[i]))

def multiclassprediction(weights, bias, data):
    outputs = []
    prediction = 0
    for i in range(len(weights)):
        sum = 0
        for j in range(len(data) - 1):
            sum += weights[i][j]*float(data[j])+bias[i]
        outputs.append(sum)

    prediction = np.argmax(outputs) + 1
    return prediction

def multiclassperceptron (train, test):
    #3 sets of weights for each case
    weights = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    bias = [0.2, 0.2, 0.2]
    lr = 0.1
    epochs = 1000

    for i in range(3):
        outputs = []
        new_train = changeclassname(train, i)
        random.shuffle(new_train)
        for data in new_train:
            if data[-1] not in outputs:
                outputs.append(data[-1])

        for epoch in range(epochs):
            for data in new_train:
                weights[i], bias[i] = training(bias[i], weights[i], data, lr, epochs, 1)

    correct  = 0
    wrong = 0
    for data in test:
        output = multiclassprediction(weights, bias, data)
        if data[-1] == "class-1":
            expected = 1
        elif data[-1] == "class-2":
            expected = 2
        elif data[-1] == "class-3":
            expected = 3
        print("actual is " +str(expected) + " predicted is " +str(output))
        if output == expected:
            correct+=1
        else:
            wrong += 1

    total = correct + wrong
    print ("accuracy is ")
    accuracy = (correct/total)*100
    print(str(accuracy))



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

def prediction(inputs, weights, bias, mode):
    sum = bias
    #len(inputs)-1 because the final element will be the output
    for i in range(len(inputs)-1):
        #weights is a 2d array
        if mode == 0:
            sum += float(weights[0][i])*float(inputs[i])
        elif mode == 1:
            sum += float(weights[i])*float(inputs[i])
    output = activation_function(sum)
    return output



#sigmoid activtation function
def activation_function(sum):
    #sigmoid activation function
    output = 1/(1 + np.exp(-sum))
    if output <= 0.5:
        return 1
    else :
        return 2

def training (bias, weights, data, lr, epochs, mode, outputs = []):

    if mode == 0:
        if data[-1] == outputs[0]:
            expected = 1
        elif data[-1] == outputs[1]:
            expected = 2
    elif mode == 1:
        if data[-1] == "class-1" or data[-1] == "class-2":
            expected = 1
        else:
            expected = 2
        '''if type(outputs) == list:
            for i in range(len(outputs)):
                if data[-1] == outputs[i]:
                    expected = i
                    '''
    #for epoch in range(epochs):

    pred = prediction(data, weights, bias, mode)
    error = expected - pred
    bias = bias + (lr * error)
    if mode == 0:
        for i in range(len(data) - 1):
            weights[0][i] = float(weights[0][i]) + (float(lr) * float(error) * float(data[i]))
        #print ("epochs = %s error = %s" %(epoch, error))
    elif mode == 1:
        '''# TODO: write if statement for weight training. That is for the correct weight, increase the weight by
                   adding weight += data. decrease the 'wrong' weight by weight -= data
        '''
        for i in range(len(data) - 1):
            weights[i] = float(weights[i]) + (float(lr) * float(error) * float(data[i]))
    return weights, bias

def regularisation (data, i):
    if i == 0:
        return data*0.01
    elif i == 1:
        return data*0.1
    elif i == 2:
        return data*0.1
    elif i == 3:
        return data*0.1
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

perceptron(train, test, weights, learning_rate, bias, epochs)
#impparameters(train, test, learning_rate, epochs)
#multiclassperceptron(train, test)
#shorten_list(train, 1)
#test_func(train)


'''
some good initial weights are:
    initial weights are [[0.43428534, 0.31845077, 0.4076814,  0.54488669]]
    initial_weights = [[0.57982218, 0.40269277, 0.48749287, 0.21941321]]
    initial_weights = [[1.06158382, -1.42830725,  0.29663148, -0.49281335]]
    final_weights = [[-0.90130712,  0.77518744,  0.6488506,   0.95299194]]

'''
