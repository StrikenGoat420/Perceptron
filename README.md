
The first 4 coloumns are the input and the final coloumn is the desired output.
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

 
