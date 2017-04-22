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

def get_data_labels(files):
    data = []
    labels = []
    #batch_size = 100
    label_count = 0
    for x, file in enumerate(files):
        temp_data = np.genfromtxt(files[0], delimiter=',')
        N = temp_data.shape[0]
        data.extend(temp_data)
        label_row = np.zeros(10)
        label_row[x] = 1
        labels.extend(np.full((N,10), label_row))

    labels = np.array(labels)
    data = np.array(data)
    return data, labels

###use testing data


training_files =['data/digit_train_0.txt', 'data/digit_train_1.txt',
'data/digit_train_2.txt', 'data/digit_train_3.txt',
'data/digit_train_4.txt', 'data/digit_train_5.txt',
'data/digit_train_6.txt', 'data/digit_train_7.txt',
'data/digit_train_8.txt', 'data/digit_train_9.txt']

testing_files = ['data/digit_test_0.txt', 'data/digit_test_1.txt',
'data/digit_test_2.txt', 'data/digit_test_3.txt',
'data/digit_test_4.txt', 'data/digit_test_5.txt',
'data/digit_test_6.txt', 'data/digit_test_7.txt',
'data/digit_test_8.txt', 'data/digit_test_9.txt']

training_data, training_labels = get_data_labels(training_files)
testing_data, testing_labels = get_data_labels(testing_files)

dummy_data = [[0, 0, 0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
dummy_labels  = [[1,0],[0,1],[0,1],[1,0],[0,1],[1,0],[1,0],[0,1]]
dummy_data = np.array(dummy_data)
dummy_labels = np.array(dummy_labels)
N = training_data.shape[0]

#training_data = create_batches(batch_size, training_data)

nn = network.neural_network((64, 32, 16, 10))
nn.train_network(training_data, training_labels, 5000, 0.1, 0.7)
nn.test_network(testing_data, testing_labels)

# for i in range(20, 64):
#     for j in range(1, 64):
#         if(i == 0):
#             nn = network.neural_network((64, j, 10))
#         else:
#             nn = network.neural_network((64, j, i, 10))
#         nn.train_network(training_data, training_labels, 1000, 0.1, 0.7)
#         nn.test_network(training_data, training_labels)

# for i in range(1, 20):
#     #for j in range(1, 64):
#     #if(i == 0):
#     nn = network.neural_network((3, i, 2))
#     # else:
#     #     nn = network.neural_network((64, j, i, 10))
#     nn.train_network(dummy_data, dummy_labels, 1000, 0.05, 0.05)
#     nn.test_network(dummy_data, dummy_labels)
