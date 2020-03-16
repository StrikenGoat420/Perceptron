import csv
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

    # TODO: Normalize our data so all the data is within the range 0 to 1
            Check if this activation function is fine or not, if not write new one
            Write "probability" function for multiclass classifier
            Automate class selection in perceptron function for binary classifier
                aka in first case ignore all class 3
                       second case ignore all class 1
                       final case ignore all class 2
'''

#this is a regular perceptron function, aka binary classifier
def perceptron(training_data, testing_data, weights, lr, bias, epochs):
    correct = 0
    wrong = 0

    print ("initial weights are ", end = '')
    print (weights)
    print ("initial bias is ", end = '')
    print (bias)
    for epoch in range(epochs):
        for key,value in enumerate(training_data):
            if training_data[key][-1] != 'class-3':
                if training_data[key][-1] == 'class-1':
                    actual_output = 1
                elif training_data[key][-1] == 'class-2':
                    actual_output = 2
                data = training_data[key]
#                pred = prediction (data, weights, bias)
                weights, bias = training(bias,weights,data,lr,epochs)
#                print("predicted output is " +str(pred) + " actual output is " +str(actual_output))
#                correct, wrong = accuracy_check(actual_output, pred, correct, wrong)

    print ("updated weights are ", end = '')
    print (weights)
    print ("updated bias is ", end = '')
    print (bias)
#    total = correct + wrong
#    accuracy = (correct/total)*100
#    print("accuracy for testing data is " +str(accuracy))

#    print("----------------------------------------------------------------------")

#    correct = 0
#    wrong = 0

    for key,value in enumerate(testing_data):
        if testing_data[key][-1] != 'class-3':
            data = testing_data[key]
            predicted_output = prediction(data, weights, bias)
            actual_output = testing_data[key][-1]
            if actual_output == 'class-1':
                actual_output = 1
            elif actual_output == 'class-2':
                actual_output = 2
            #print("predicted output is " +str(predicted_output) + " actual output is " +str(actual_output))
            correct, wrong = accuracy_check(actual_output, predicted_output, correct, wrong)

    total = correct + wrong
    accuracy = (correct/total)*100
    print("accuracy for testing data is " +str(accuracy))


def percforimppara(training_data, testing_data, weights, lr, bias, epochs):
    correct = 0
    wrong = 0

    print ("initial weights are ", end = '')
    print (weights)
    for epoch in range(epochs):
        print('a')
        if training_data[-1] != 'class-3':
            if training_data[key][-1] == 'class-1':
                actual_output = 1
            elif training_data[key][-1] == 'class-2':
                actual_output = 2
            weights, bias = training(bias,weights, training_data, lr, epochs)

    print("weights after training are ")
    print(weights)

    '''if testing_data[-1] != 'class-3':
        predicted_output = prediction(testing_data, weights, bias)
        actual_output = testing_data[-1]
        if actual_output == 'class-1':
            actual_output = 1
        elif actual_output == 'class-2':
            actual_output = 2
        print("predicted output is " +str(predicted_output) + " actual output is " +str(actual_output))
        correct, wrong = accuracy_check(actual_output, predicted_output, correct, wrong)

    total = correct + wrong
    accuracy = (correct/total)*100
    print("accuracy is " +str(accuracy))
    '''


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


    #for data in train:
    #    print(data)

def test_func (train):
    print('a')


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

def training (bias, weights, data, lr, epochs):
    expected = 0
    if data[-1] == "class-1":
        expected = 1
    elif data[-1] == "class-2":
        expected = 2

    #for epoch in range(epochs):

    pred = prediction(data, weights, bias)
    error = expected - pred
    bias = bias + (lr * error)
    for i in range(len(data) - 1):
        weights[0][i] = float(weights[0][i]) + (float(lr) * float(error) * float(data[i]))
    #print ("epochs = %s error = %s" %(epoch, error))

    return weights, bias


# importing data
with open ('train.data', 'r') as f:
    train = list(csv.reader(f,delimiter=','))

with open ('test.data', 'r') as f:
    test = list(csv.reader(f,delimiter=','))


train = list(np.array(train, dtype=None))
test = list(np.array(test, dtype=None))

#random.shuffle(train)

bias = 0.2
#weights = [[0,0,0,0]]
weights = np.random.rand(1,4)
learning_rate = 0.1
epochs = 20

#perceptron(train, test, weights, learning_rate, bias, epochs)
impparameters(train, test, learning_rate, epochs)
#test_func(train)


'''
some good initial weights are:
    initial weights are [[0.43428534, 0.31845077, 0.4076814,  0.54488669]]
    initial_weights = [[0.57982218, 0.40269277, 0.48749287, 0.21941321]]
    initial_weights = [[1.06158382, -1.42830725,  0.29663148, -0.49281335]]
    final_weights = [[-0.90130712,  0.77518744,  0.6488506,   0.95299194]]

'''
