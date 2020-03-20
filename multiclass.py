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

def multiclassprediction(weight, bias, data):
    ''' weights has to a a 2d array, consisting of weights for all 3 cases. Bias also has to
        to have all 3 biases for all cases'''
    outputs = []
    for i in range(len(weight)):
        pred = 0
        for j in range(len(data) - 1):
            pred += weight[i][j] * data[j] + bias[j]
        outputs.append(pred)
        pred = 0

    output = np.argmax(outputs)
    return output


def multiclassperceptron (train, test):
    #3 sets of weights for each case
    weights = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    bias = [0.2, 0.2, 0.2]
    lr = 0.1
    '''What we have to do?
       create a training function which accepts 2d arrays of weights, and an array for the bias.
       After accepting the weights and the bias, as well as the dataset (which is initially unmodified ie.
       not isolated), for weights[i] and bias [i] we have to isolate the appropriate class. After isolating
       the class, we send the whole weights and the bias into the predict function. Over at the predict
       function, it accepts
    '''

    for i in range(3):
        outputs = []
        case = i
        new_trainingdata = changeclassname(train, case)
        for data in new_trainingdata:
            if data[-1] not in outputs:
                outputs.append(data[-1])

        for data in new_trainingdata:
            weights[i], bias[i] = multiclasstraining(bias[i], weights[i], data, outputs, lr)
        outputs = []

    print("training is done :D ")
    print("new weights are ")
    for data in weights:
        print(data)

    correct = 0
    wrong = 0
    probability = []
    actual_prediction = 0
    for data in test:
        for i in range(3):
            probability.append(multiclassprediction(data, weights[i], bias[i]))
            if len(probability) == 3:
                #print("probabilities are ")
                #print(probability)
                actual_prediction = max(probability)
                if actual_prediction == probability[0]:
                    actual_prediction = "class-3"
                elif actual_prediction == probability[1]:
                    actual_prediction = "class-2"
                elif actual_prediction == probability[2]:
                    actual_prediction = "class-1"
            actual_output = data[-1]
            print("prediction is " +str(actual_prediction)+ " actual output is " +str(actual_output))
            print("--------------")
            correct, wrong = accuracy_check(actual_output, actual_prediction, correct, wrong)
        probability = []
