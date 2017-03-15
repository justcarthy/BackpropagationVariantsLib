'''
This is the opening class of the neural network which performs these tasks
- Reads the input data from their respective files (X)
- Randomly shuffles the list (X)
- Splits list into minibatches of specified size
- then procedes to create the neural network based on its specified layer and
    hidden node count
'''
import numpy as np
import network

def create_batches(size, data):
    N = data.shape[0]
    i = 0
    batched_data = []
    while i + size <= N:
        batched_data.append(data[i:i+size, :])
        i += size
    return np.array(batched_data)


training_files =['data/digit_train_0.txt', 'data/digit_train_1.txt',
'data/digit_train_2.txt', 'data/digit_train_3.txt',
'data/digit_train_4.txt', 'data/digit_train_5.txt',
'data/digit_train_6.txt', 'data/digit_train_7.txt',
'data/digit_train_8.txt', 'data/digit_train_9.txt']

dummy_data = [[0, 0, 0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
dummy_labels  = [[1,0],[0,1],[0,1],[1,0],[0,1],[1,0],[1,0],[0,1]]
dummy_data = np.array(dummy_data)
dummy_labels = np.array(dummy_labels)
training_data = []
training_labels = []
#batch_size = 100
label_count = 0
for x, file in enumerate(training_files):
    temp_data = np.genfromtxt(training_files[0], delimiter=',')
    N = temp_data.shape[0]
    training_data.extend(temp_data)
    label_row = np.zeros(10)
    label_row[x] = 1
    training_labels.extend(np.full((N,10), label_row))
    
training_labels = np.array(training_labels)
training_data = np.array(training_data)
N = training_data.shape[0]

#training_data = create_batches(batch_size, training_data)

nn = network.neural_network((64, 32, 16, 10))
nn.train_network(training_data, training_labels, 10000, 0.4)
nn.test_network(training_data, training_labels)
