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
        return 1 / (1 + np.exp(-1*inputs))
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
            self.weights = np.random.uniform(-0.75, 0.75,layer_shape)
            self.weight_updates = np.zeros(layer_shape)
        if not b_output and not b_input:
            self.error_signal = np.zeros(layer_shape[0])
        print(self.weights, self.error_signal)

    def feed_forward(self):
        if self.input_layer:
            self.output = self.inputs.dot(self.weights)
        elif self.output_layer:
            self.output = sigmoid(self.inputs)
            self.error_signal = (sigmoid(self.inputs, derivative=True))
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
            total_error = 0.0
            error = None
            network_error = 0.0
            randomRange = list(range(data.shape[0]))
            random.shuffle(randomRange)
            for i in randomRange:
                output = self.forward_propogate(data[i])
                error = output - labels[i]
                #if(self.is_error(output, labels[i])):
                self.back_propogate(error)
                self.update_weights(learning_rate)
                total_error = (np.sum(0.5*(np.square(error))))
                network_error += total_error
            print('epoch: {} error is {}'.format(e, network_error/data.shape[0]))


    def is_error(self, output, actual):
        if (np.argmax(output) != np.argmax(actual)):
            return True
        else:
            return False

    def forward_propogate(self, data):
        
        self.layers[0].inputs = data
        for j in range(self.num_layers-1):
            self.layers[j+1].inputs = self.layers[j].feed_forward()
        
        return self.layers[-1].feed_forward()

    def back_propogate(self, error):
        self.layers[-1].error_signal = (error)*self.layers[-1].error_signal
        for j in range(self.num_layers-2, 0, -1):
            weights = self.layers[j].weights
            self.layers[j].error_signal = weights.dot(self.layers[j+1].error_signal)*self.layers[j].error_signal

    def update_weights(self, learning_rate):
        for i in range(self.num_layers-1):
            gradient_update = (-learning_rate)*self.layers[i+1].error_signal.dot(self.layers[i].output)
            self.layers[i].weights += gradient_update

    def test_network(self, data, labels):
        randomRange = list(range(data.shape[0]))
        random.shuffle(randomRange)
        correct = 0
        wrong = 0
        for i in randomRange:
            #forward prop
            output = self.forward_propogate(data[i])
            value = np.argmax(output)
            actual= np.argmax(labels[i])
            print('output:{} ..... actual:{}'.format(value, actual))
            if(value == actual):
                correct+=1

        print('{} out of {}'.format(correct, data.shape[0]))




