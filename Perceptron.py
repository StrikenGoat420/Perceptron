import csv
import numpy as np

'''The first 4 coloumns are the input and the final coloumn is the desired output.
    Random weights have been assigned in the main function. Data will go one row at a time into
    the training function, this is where the weight updation will be happening based on the prediction.
    The perceptron function is the one containing the actual perceptron algorithm. The output of which
    depends on the activation function.
    # TODO: Normalize our data so all the data is within the range 0 to 1
            Convert our output to an integer
            (maybe if output == 0, then print(class 0) and if output == 1 print (class 1))
            Write a proper activation function (gradient descent)

    PS: The clean_list function is really badly writted, write it properly when possible
'''

def perceptron1(inputs, weights, bias):
    sum = bias
    #len(inputs)-1 because the final element will be the output
    for i in range(len(inputs)-1):
        #weights is a 2d array
        sum += float(weights[0][i])*float(inputs[i])
    output = activation_function(sum)
    return output

def activation_function(sum):
    #temporary activation function
    if (sum > 0 ):
        return sum
    else : return 0


def training (bias, weights, data, lr, epochs):
    expected = 0
    if data[-1] == "class-1":
        expected = 1
    elif data[-1] == "class-2":
        expected = 2

    for epoch in range(epochs):
        pred = perceptron1(data, weights, 0.2)
        print ("class = %d expected = %d predicted = %d" %(data[-1],expected, pred))
        error = expected - pred
        bias = bias + (lr * error)
        for i in range(len(data) - 1):
            weights[0][i] = float(weights[0][i]) + (float(lr) * float(error) * float(data[i]))
        print ("epochs = %s error = %s" %(epoch, error))


#function to remove class-3 from training and testing set cuz its not needed
def clean_list(list):
    for key, value in enumerate(list):
        try:
            if 'class-3' in value:
                list = list.remove(key)
        except AttributeError and ValueError:
            rando = 0
    return list


# importing data
with open ('train.data', 'r') as f:
    train = list(csv.reader(f,delimiter=','))

with open ('test.data', 'r') as f:
    test = list(csv.reader(f,delimiter=','))

train = list(np.array(train, dtype=None))
test = list(np.array(test, dtype=None))
train = clean_list(train)
test = clean_list(test)

bias = 0.2
weights = np.random.rand(1,4)
learning_rate = 0.1
epochs = 1

for key, value in enumerate(train):
    data = train[key]
    training(bias,weights,data, learning_rate, epochs)

#training(inputs, target, 0.1)


lr = 3
