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
        return 0.1*sigmoid(inputs)*(1 - sigmoid(inputs))

class network_layer:
    """
    each layer holds:
        - input array
        - weight matrix
        - output array
        - weight deltas for updating weights on gradientogation
        - and the sigmoid derivative values for this update
    """
    
    def __init__(self, layer_shape, batch_size=1, b_input=False, b_output=False ):
        self.input_layer = b_input
        self.output_layer = b_output
        self.previous_weights = np.zeros(layer_shape)
        self.previous_error = np.zeros(layer_shape)
        self.update_value = np.zeros(layer_shape)
        self.inputs = None
        self.weights = None
        self.outputs = None
        self.activated = None
        self.error_signal = None #
        self.derivative = None #derivative of sigmoid functions
        #input layers do not need their inputs initialized
        if not b_input:
            self.inputs = np.zeros(layer_shape[0])
            #self.error_signal = np.zeros(layer_shape)
        if not b_output:
            self.weights = np.random.normal(size=layer_shape, scale=0.5)
            self.error_signal = np.zeros(layer_shape)


    '''the feed forward has 3 cases, input layer, output layer or hidden layer
    - input is the basic multiplication by weights, and appending the bias node
    - hidden is activating the inputs, getting the derivative of this activation, then multiply by weights
    - output layer is activating, getting the derivative, then output the result''' 
    def feed_forward(self):
        if self.input_layer:
            self.activated = np.append(self.inputs, np.ones((self.inputs.shape[0], 1)), axis=1)
            self.output = self.activated.dot(self.weights)
        elif self.output_layer:
            self.output = self.activated = sigmoid(self.inputs)
            self.derivative = (sigmoid(self.inputs, derivative=True)).T
        else:
            #adding bias value
            self.activated = np.append(sigmoid(self.inputs), np.ones((self.inputs.shape[0], 1)), axis=1)
            self.output = self.activated.dot(self.weights)
            self.derivative = (sigmoid(self.inputs, derivative=True)).T
        return self.output

'''The neural network class, initialized with the general shape, and batch size
The shape is then used to create the appropriate layers with bias, identifying input/output'''
class neural_network:
    def __init__(self, network_shape, batch_size=1):
        self.num_layers = len(network_shape)
        self.batch_size = batch_size
        self.network_shape = network_shape
        self.layers = []
        for x in range(self.num_layers):
            #do the +1 on the layer nodes to represent the biases
            if x == 0: # x is an input layer
                self.layers.append(network_layer((network_shape[x]+1, network_shape[x+1]), b_input=True))
            elif x == (self.num_layers - 1): # x is an output layer
                self.layers.append(network_layer((network_shape[x], 1), b_output=True))
            else:
                self.layers.append(network_layer((network_shape[x]+1, network_shape[x+1])))
    
    '''train_network first creates the batches of input data, and initializes any recording variables'''
    def train_network(self, data, labels, epochs, update, *args):
        data, labels = create_batches(self.batch_size, data, labels)
        epoch_result = []
        for e in range(epochs):
            '''total error, and network error used for measurement'''
            total_error = 0.0
            error = None
            network_error = 0.0
            '''randomRange is created and iterated through to have a random iteration of the data [5,1,6,4...]'''
            randomRange = list(range(data.shape[0]))
            random.shuffle(randomRange)
            for i in randomRange:
                output = self.forward_propogate(data[i])
                error = (output - labels[i])
                self.back_propogate(error)
                '''manually choosing the weight update method'''
                '''since this was for testing, many variables were simply hardcoded'''
                update(*args)
                #self.quickprop_update(q_learn_rate, g_learn_rate, momentum)
                #self.gradient_update(g_learn_rate, momentum)
                #self.rprop_update()
                #self.delta_bar_update()
                '''used to calculate mse'''
                total_error = np.sum(np.square(error))/self.batch_size
                network_error += total_error
            mse = network_error/data.shape[0]
            if mse < 0.001:
                print 'converged'
                break
            print('epoch: {} error is {}'.format(e, mse))
            epoch_result.append([args, e, mse])
        print "quick learning rate {}, gradient learning rate {}, momentum {}".format(q_learn_rate, g_learn_rate, momentum)
        return epoch_result
        
    def is_error(self, output, actual):
        if (np.argmax(output) != np.argmax(actual)):
            return True
        else:
            return False
    '''feed forward on the network level'''
    def forward_propogate(self, data):
        # TODO: This is where the check for batch size == 1 should go
        self.layers[0].inputs = data

        for j in range(self.num_layers-1):
            self.layers[j+1].inputs = self.layers[j].feed_forward()
        
        return self.layers[-1].feed_forward()
    '''backpropagation, error gradient calculated iterating backwards through layers
    error_signal in each layer is the error gradient used to update each weight'''
    def back_propogate(self, error):
        self.layers[-1].error_signal = (error.T)
        for j in range(self.num_layers-2, 0, -1):
            weights = self.layers[j].weights[0:-1, :]
            self.layers[j].error_signal = (weights).dot(self.layers[j+1].error_signal)*self.layers[j].derivative
    '''The standard update function for back propagation
    - error is calculated for each weight when multiplied by the activation function
    - momentum is applied to the update along with with the learning rate''' 
    def gradient_update(self, learning_rate, momentum):
        for i in range(0, self.num_layers-1):
            error = (self.layers[i+1].error_signal.dot(self.layers[i].activated)).T
            gradient_update = (-learning_rate)*error + (momentum)*self.layers[i].previous_weights
            self.layers[i].weights += gradient_update
            self.layers[i].previous_weights = gradient_update
 
    '''QuickProp is finnicky and doesnt work on some datasets,
    first, I perform a check to see if the difference between the current and 
    previous errors are not 0, to avoid the division by 0, if so I skip that step
    (curr_err/ prev_err-curr_err)
    and go straight to the gradient step, followed by the same sign increse,
    the maximum growth function, and finally the weight decay, exactly as in the paper'''
    def quickprop_update(self, q_learn_rate, g_learn_rate, momentum):
        max_growth = 1.75
        weight_decay = 1e-3

        for i in range(0, self.num_layers-1):
            if np.all(self.layers[i].previous_weights == 0):
                self.layers[i].previous_weights = np.ones(self.layers[i].weights.shape)
            previous_error = self.layers[i].previous_error
            previous_weights = self.layers[i].previous_weights
            weights = self.layers[i].weights
            '''current error calculated'''
            current_error = (self.layers[i+1].error_signal.dot(self.layers[i].activated)).T
            '''difference check'''
            difference = (previous_error - current_error)
            delta_weight = np.zeros(weights.shape)
            if np.any(difference == 0):
                delta_weight = np.zeros(weights.shape)
            else:
                '''Newtonian function of minimizing parabola'''
                delta_weight = previous_weights*(np.divide(current_error, difference))
            '''if the current and previous error signs are the same, this is a good direction
            slight increase by the current error'''
            delta_weight[current_error*previous_error >= 0] \
                += (-q_learn_rate)*current_error[current_error*previous_error >= 0]
            '''Gradient step is added on every pass'''
            delta_weight += (-g_learn_rate)*current_error + (momentum)*previous_weights
            '''if the current increase of a weight is greater than 1.75 the previous weight it gets cut off at that'''
            delta_weight[(current_error*previous_error >= 0) & (np.abs(delta_weight) > np.abs(previous_weights*max_growth))] \
                = max_growth*previous_weights[(current_error*previous_error >= 0) \
                & (np.abs(delta_weight) > np.abs(previous_weights*max_growth))]
        
            delta_weight += previous_weights*weight_decay

            weights += delta_weight
            self.layers[i].weights = weights
            self.layers[i].previous_weights = delta_weight
            self.layers[i].previous_error = current_error
    
    '''rprop update is used for an update value, similar to learning rate, but uses no gradient step
    only the learning rate'''
    def rprop_update(self):
        '''increase and decrease of weights by these factors'''
        npos, nneg = 1.2, 0.5
        '''dmax, dmin, are the max and min a weight can change'''
        dmax, dmin = 50.0, 1e-6
        for i in range(0, self.num_layers-1):
            previous_error = self.layers[i].previous_error
            update_value = self.layers[i].update_value
            weights = self.layers[i].weights
            delta_weight = np.zeros(weights.shape)
            current_error = (self.layers[i+1].error_signal.dot(self.layers[i].activated)).T
            '''comparing previous error sign with current'''
            ## Values with a sign +1
            update_value[current_error*previous_error > 0] = update_value[current_error*previous_error > 0]*npos
            update_value[update_value > dmax] = dmax
            delta_weight[current_error*previous_error > 0] = update_value[current_error*previous_error > 0]\
                *np.sign(current_error[current_error*previous_error > 0])
            ## Values with sign 0
            delta_weight[current_error*previous_error == 0] = update_value[current_error*previous_error == 0]\
                *np.sign(current_error[current_error*previous_error == 0])
            ## Values with sign -1
            update_value[current_error*previous_error < 0] = update_value[current_error*previous_error < 0]*nneg
            update_value[update_value < dmin] = dmin
            current_error[current_error*previous_error < 0] = 0
            delta_weight[current_error*previous_error < 0] = update_value[current_error*previous_error < 0]
            ## resetting values
            weights -= delta_weight
            self.layers[i].previous_error = current_error
            self.layers[i].update_value = update_value
            self.layers[i].weights = weights

    ''''Delta bar-Delta,
        somewhat blend of backprop and rprop, where the learning rate is independent for each weight'''            
    def delta_bar_update(self):
        b = 0.01
        k = 0.01
        decay = 0.001
        
        for i in range(0, self.num_layers-1):
            previous_delta = self.layers[i].previous_error
            delta = (self.layers[i+1].error_signal.dot(self.layers[i].activated)).T
            '''bar_delta is the average of the current and previous deltas'''
            bar_delta = (1-b)*delta + b*previous_delta
            '''the learning rate'''
            update_value = self.layers[i].update_value
            '''initializing the learn_rates'''
            if(np.all(update_value == 0)):
                update_value = np.full(delta.shape, 0.1)
            '''if signs are the same'''
            update_value[delta * previous_delta > 0] = update_value[delta * previous_delta > 0]
            '''signs are different'''
            update_value[delta * previous_delta < 0] = update_value[delta * previous_delta < 0]*(1-decay)
            #print update_value
            '''previous update_value used if 0'''
            self.layers[i].weights -= update_value*delta
            self.layers[i].previous_error = (bar_delta*delta)/2
            self.layers[i].update_value = update_value



    def test_network(self, data, labels):
        data, labels = create_batches(1, data, labels)
        correct = 0
        wrong = 0
        for i in range(data.shape[0]):
            #forward prop
            output = self.forward_propogate(data[i])
            #uses argmax to get answer
            value = np.argmax(output)
            actual= np.argmax(labels[i])
            if(value == actual):
                correct+=1
        percent = (float(correct)/data.shape[0])*100
        print('{:3.2f} % accuracy with {} '.format((float(correct)/data.shape[0])*100, self.network_shape))
        return [[percent, self.network_shape, self.batch_size]]

def create_batches(size, data, labels):
    data_shuf = []
    labels_shuf = []
    randomRange = list(range(data.shape[0]))
    random.shuffle(randomRange)
    for i in randomRange:
        data_shuf.append(data[i])
        labels_shuf.append(labels[i])
    N = data.shape[0]
    i = 0
    batched_data = []
    batched_labels = []
    while i + size <= N:
        batched_data.append(data_shuf[i:i+size])
        batched_labels.append(labels_shuf[i:i+size])
        i += size
    return np.array(batched_data), np.array(batched_labels)





