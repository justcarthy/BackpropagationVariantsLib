'''
This is the opening class of the neural network which performs these tasks
- Reads the input data from their respective files (X)
- Randomly shuffles the list (X)
- Splits list into minibatches of specified size
- then procedes to create the neural network based on its specified layer and
    hidden node count
'''
import itertools

import numpy as np
import csv
from sklearn import datasets
import network

def get_training_testing(data, labels):
    withheld_amt = int(.15 * len(data))
    testing_data = []
    testing_labels = []
    training_data = np.array(data, dtype='float32')
    training_labels = np.array(labels,dtype='float32')
    for i in range(withheld_amt):
        index = np.random.randint(len(training_data))
        testing_data.append(training_data[index])
        testing_labels.append(training_labels[index])
        training_data = np.delete(training_data, index, axis=0)
        training_labels = np.delete(training_labels, index, axis=0)
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)
    return training_data, training_labels, testing_data, testing_labels

'''iris data'''
# iris = datasets.load_iris()
# x = iris.data
# temp = iris.target
# y = []
# for label in temp:
#     if(label == 0):
#         y.append([0,0,1])
#     elif (label == 1):  
#         y.append([0,1,0])
#     else:
#         y.append([1,0,0])
# training_data, training_labels, testing_data, testing_labels = get_training_testing(x, y)
'''breast cancer data'''
with open('data/wdbc.data', 'rb') as f:
    reader = csv.reader(f)
    wdbc = list(reader)
wdbc_temp = [row[1] for row in wdbc]
wdbc = [row[2:] for row in wdbc]
wdbc_labels = []
for label in wdbc_temp:
    if(label == 'M'):
        wdbc_labels.append([0,1])
    else:
        wdbc_labels.append([1,0])
training_data, training_labels, testing_data, testing_labels = get_training_testing(wdbc, wdbc_labels)
'''wine data'''
# with open('data/wine.data', 'rb') as f:
#     reader = csv.reader(f)
#     wine = list(reader)
# wine_temp = [row[0] for row in wine]
# wine = [row[1:] for row in wine]
# wine_labels = []
# for label in wine_temp:
#     if(label == 0):
#         wine_labels.append([0,0,1])
#     elif(label == 1):
#         wine_labels.append([0,1,0])
#     else:
#         wine_labels.append([1,0,0])
# training_data, training_labels, testing_data, testing_labels = get_training_testing(wine, wine_labels)
'''create combination of weights'''
g_learn_rates = [0.01, 0.1, 0.4]
q_learn_rates = [0.1, 0.9, 1.5]
momentum = [0.01, 0.05, 0.1]
batch = [training_data.shape[0]/4, training_data.shape[0]/2, training_data.shape[0]]
combinations = itertools.product(q_learn_rates, g_learn_rates, momentum, batch)
for run in combinations:
    result = []
    test = []
    ''' nn represents the quickprop learning rates, the gradient update learning rates,
        momentum, and delta bar delta variables, as the args param. I know this is bad'''
    ''' gradient_update(learning_rt, momentum)
        quickprop_update(q_learning_rt, g_learning_rt, momentum)
        r_prop()
        delta_bar_update()'''
    nn = network.neural_network((training_data.shape[1], 20, 20, training_labels.shape[1]), run[3])#training_data.shape[0]/2)
    result = nn.train_network(training_data, training_labels, 3000, nn.rprop_update)
    test = nn.test_network(testing_data, testing_labels)
    with file('res.txt', 'a') as outfile:
            for item in result:
                outfile.write("{}\n".format(item))
    with file('test.txt', 'a') as outfile:
            for item in test:
                outfile.write("{}\n".format(item))