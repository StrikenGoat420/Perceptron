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
    print("inputs in perceptron1 is " +str(inputs))
    print("weights in perceptron1 is " +str(weights))
    #len(inputs)-1 because the final element will be the output
    for i in range(len(inputs)-1):
        sum += weights[i]*2
        #sum += float(weights[i])*float(inputs[i])
        #sum += float(weights[i])*float(inputs[i])
    output = activation_function(sum)
    return output

def activation_function(sum):
    #temporary activation function
    if (sum > 0 ):
        return sum
    else : return 0


def training (bias, weights, data, lr, epochs):
    for epoch in range(epochs):
        pred = perceptron1(data, weights, 0.2)
        error = row[-1] - pred
        bias = bias + (lr * error)
        for i in range(len(data-1)):
            weights[i] = weights[i] + (lr * error * data[i])
        print ("%s epochs %s error", epoch, error)


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
epochs = 20

for key, value in enumerate(train):
    data = train[key]
    training(bias,weights,data, learning_rate, epochs)

#training(inputs, target, 0.1)


lr = 3
