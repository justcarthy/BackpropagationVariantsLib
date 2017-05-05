# Backpropagation Variants Library

This is a Backpropagation variants library used for testing on various datasets. The variants include:
1. Gradient Descent
  * Based on the standard backpropagation algorithm, using the weights contribution to error
  via slope to determine the weight delta.
2. QuickProp
  - Based on [this paper](http://repository.cmu.edu/cgi/viewcontent.cgi?article=2799&context=compsci) by Scott Fahlman
3. R-Prop
  - Each weight consists of an independent learning rate, updated based on previous delta weights. [Reference](http://www.inf.fu-berlin.de/lehre/WS06/Musterererkennung/Paper/rprop.pdf).
4. Delta Bar-Delta
  - Similar to rprop, but update also consists of a gradient descent. [Reference](https://www.willamette.edu/~gorr/classes/cs449/Momentum/deltabardelta.html)
  
## Using the library
- The network library is imported into the python file. 
`import network`

- Create the neural network object.
```python
nn = network.neural_network((input_size, hidden1_size, hidden2_size, output_size), batch_size)
```
- Calling the train_network method, will train the weights on the provided training data and labels, # of epochs,
as well as the update function specified
- With this the update function requires the specific parameters:
1. `gradient_update(learning_rt, momentum)`
2. `quickprop_update(q_learning_rt, g_learning_rt, momentum)`
- Returned from the train_network is a list of the epochs and their mean squared error
```python
result = nn.train_network(train_data, train_labels, epochs, nn.rprop_update, ...)
```

- To test the model, test_network is called, returning the accuracy of the network.
```python
test = nn.test_network(testing_data, testing_labels)
```

### Improvements
So far this has only been used for testing on a school project, but I do hope to expand this to provide more functionality, rather than hardcoding learning rates,
And one-off use of the model.
