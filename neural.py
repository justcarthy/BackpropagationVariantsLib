#initialize the weights of the neural network
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inputs, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-inputs))
    else:
        return sigmoid(inputs)*(1 - sigmoid(inputs))


X_file = np.genfromtxt('data/digit_test_0.txt', delimiter=',')

N, F = np.shape(X_file)
#hidden layer nodes
H = 5

train_set = X_file[0:199]
test_set = X_file[200:400]


layer1_W = np.random.normal(size=[F, H], scale=1.0)
i = 0
x = train_set[0] #for x in train_set:
S = x.dot(layer1_W)
Z = sigmoid(S)
layer_output_W = np.random.normal(size=[layer1_W.shape[0]])
print(Z)



    # = x.dot(layer1_W)



#print(out)
