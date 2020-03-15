import csv
import numpy as np

'''The first 4 coloumns are the input and the final coloumn is the desired output.
    Random weights have been assigned in the main function. Data will go one row at a time into
    the training function, this is where the weight updation will be happening based on the prediction.
    The perceptron function is the one containing the actual perceptron algorithm. The output of which
    depends on the activation function.

    We are using the sigmoid activation function, because ours is a binary classifier. The sigmoid function
    will give output in the range of 0 to 1. In our case if the output is less then 0.5, prediction will be
    1 and if the output is more than 0.5 prediction will be 2.

    # TODO: Normalize our data so all the data is within the range 0 to 1
            Check if this activation function is fine or not, if not write new one.
            Better way to analyse the output of the perceptron
            The clean_list function is really badly written, write it properly if possible
'''

def perceptron(training_data, testing_data, weights, lr, bias, epochs):
    correct = 0
    wrong = 0
    print ("initial weights are ", end = '')
    print (weights)
    for epoch in range(epochs):
        for key,value in enumerate(training_data):
            if training_data[key][-1] != 'class-3':
                data = train[key]
                weights, bias = training(bias,weights,data,lr,epochs)
    print("after training weights are ", end = '')
    print(weights)

    for key,value in enumerate(testing_data):
        if testing_data[key][-1] != 'class-3':
            data = testing_data[key]
            predicted_output = prediction(data, weights, bias)
            actual_output = testing_data[key][-1]
            if actual_output == 'class-1':
                actual_output = 1
            elif actual_output == 'class-2':
                actual_output = 2
            print("predicted output is " +str(predicted_output) + " actual output is " +str(actual_output))
            correct, wrong = accuracy_check(actual_output, predicted_output, correct, wrong)

    total = correct + wrong
    accuracy = (correct/total)*100
    print("accuracy for testing data is " +str(accuracy))


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


#function to remove class-3 from training and testing set cuz its not needed
def clean_list(list):
    for key, value in enumerate(list):
        try:
            if 'class-3' == list[key][-1]:
                print('yes')
                list = list.remove(key)
        except AttributeError and ValueError:
            rando = 0
    for data in list:
        print(data)
    return list


# importing data
with open ('train.data', 'r') as f:
    train = list(csv.reader(f,delimiter=','))

with open ('test.data', 'r') as f:
    test = list(csv.reader(f,delimiter=','))

train = list(np.array(train, dtype=None))
test = list(np.array(test, dtype=None))

bias = 0.2
#weights = [[0,0,0,0]]
weights = np.random.rand(1,4)
learning_rate = 0.1
epochs = 20

perceptron(train, test, weights, learning_rate, bias, epochs)
lr = 3

'''
some good initial weights are:
    initial weights are [[0.43428534, 0.31845077, 0.4076814,  0.54488669]]
    initial_weights = [[0.57982218, 0.40269277, 0.48749287, 0.21941321]]
    initial_weights = [[1.06158382, -1.42830725,  0.29663148, -0.49281335]]
    final_weights = [[-0.90130712,  0.77518744,  0.6488506,   0.95299194]]

'''
