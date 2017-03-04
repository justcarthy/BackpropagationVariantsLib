import numpy as np
import random
"""
this is the neural network framework, neural network stores:
- the entire configuration of the network(layers, defining input and output)
- process of training the network
- testing the network, and finally output of the model

TODO change to support both minibatches and online
"""
def sigmoid(inputs, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-inputs))
    else:
        return sigmoid(inputs)*(1 - sigmoid(inputs))

class network_layer:
    """
    each layer holds:
        - input array
        - weight matrix
        - output array
        - weight deltas for updating weights on backpropogation
        - and the sigmoid derivative values for this update
    """
    def __init__(self, layer_shape, batch_size=1, b_input=False, b_output=False ):
        self.input_layer = b_input
        self.output_layer = b_output
        self.inputs = None
        self.weights = None
        self.outputs = None
        self.weight_updates = None #
        self.error_signal = None #derivative of sigmoid functions
        #input layers do not need their inputs initialized
        if not b_output:
            self.weights = np.random.normal(size = layer_shape, scale=1.0)
            self.weight_updates = np.zeros(layer_shape)
        if not b_output and not b_input:
            self.error_signal = np.zeros(layer_shape[0])

    def feed_forward(self):
        if self.input_layer:
            self.output = self.inputs.dot(self.weights)
        elif self.output_layer:
            self.output = sigmoid(self.inputs)
        else:
            self.output = (sigmoid(self.inputs)).dot(self.weights)
            self.error_signal = (sigmoid(self.inputs, derivative=True))
        return self.output

class neural_network:
    def __init__(self, network_shape, batch_size=1):
        self.num_layers = len(network_shape)
        self.layers = []
        for x in range(self.num_layers):
            #do the +1 on the layer nodes to represent the biases
            if x == 0: # x is an input layer
                self.layers.append(network_layer((network_shape[x], network_shape[x+1]), b_input=True))
            elif x == (self.num_layers - 1): # x is an output layer
                self.layers.append(network_layer((network_shape[x], 1), b_output=True))
            else:
                self.layers.append(network_layer((network_shape[x], network_shape[x+1])))
    def train_network(self, data, labels, epochs, learning_rate):
        for e in range(epochs):
            #iterate through data random
            error = 0.0
            randomRange = list(range(data.shape[0]))
            random.shuffle(randomRange)
            for i in randomRange:
                print(data[i])
                output = self.forward_propogate(data[i])
                print (output)
                error = output - labels[i];
                if(is_error(output, labels[i])):
                    self.back_propogate(error)
                    self.update_weights(learning_rate)
                    mse = (np.sum(np.square(error)))/labels.shape[1]
                    error += mse
            print('epoch: {} error is {}'.format(e, error/data.shape[0]))


    def is_error(output, actual):
        if (np.argmax(output) != np.argmax(actual)):
            return True
        else:
            return False

    def forward_propogate(self, data):
        self.layers[0].inputs = data
        for j in range(self.num_layers-1):
            self.layers[j+1].inputs = self.layers[j].feed_forward()
        return self.layers[-1].outputs

    def back_propogate(self, error):
        self.layers[-1].error_signal = (error)
        for j in range(self.num_layers-1, 0, -1):
            weights = self.layers[j].weights
            self.layers[j].error_signal = weights.dot(self.layers[j+1].error_signal)*self.layers[j].error_signal

    def update_weights(self, learning_rate):
        for i in range(self.num_layers-1):
            gradient_update = (-learning_rate)*self.layer[i+1].error_signal.dot(self.layer[i].output)
            self.layers[j].weights += gradient_update

    def test_network(self, data, labels):
        randomRange = list(range(data.shape[0]))
        random.shuffle(randomRange)
        correct = 0
        wrong = 0
        for i in randomRange:
            #forward prop
            for j in range(self.num_layers-1):
                self.layers[j+1].inputs = self.layers[j].feed_forward()
            output = self.layers[-1].outputs
            value = np.argmax(output)
            actual= np.argmax(labels[i])
            if(value == actual):
                correct+=1
        #print(correct/data.shape[0])
