import numpy as np
import h5py
import matplotlib.pyplot as plt


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network    

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1]) # * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters;

def sigmoid(X):
    Y = 1 / (1 + np.exp(-X))    
    return Y


def dsigmoid(X):
    S = sigmoid(X)
    dS = S * (1 - S)
    return dS

def relu(X):
    Y = np.maximum(0, X)
    return Y

def drelu(X):
    dR = np.array(X, copy = True)
    dR[X <= 0] = 0
    dR[X > 0] = 1

    return dR

def forwardpropagation(X, parameters):

    m = X.shape[1]              # number of examples
    n = X.shape[0]              # number of input units
    L = len(parameters) // 2    # number of layers (must // 2 because parameters contains 'W' e 'b')

    cache = {}

    A = X
    cache["A0"] = A

    # hidden layers
    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A) + b
        A = relu(Z)
        cache["A" +  str(l)] = A
        cache["Z" +  str(l)] = Z

    # output layer
    W = parameters["W" + str(l + 1)]
    b = parameters["b" + str(l + 1)]
    Z = np.dot(W, A) + b
    A = sigmoid(Z)
    cache["A" +  str(l + 1)] = A
    cache["Z" +  str(l + 1)] = Z

    Yhat = A

    return Yhat, cache

def compute_cost(Y, Yhat):
    """
    Implement the cost function

    Arguments:
    Yhat -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]      # number of examples
    cost = - (1 / m) * np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat), axis = 1, keepdims = True)
    cost = np.squeeze(cost)
    return cost

def backpropagation(Y, Yhat, cache, parameters):
    """
    Implement back-propagation

    Arguments:
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    Yhat -- probability vector corresponding to your label predictions, shape (1, number of examples)
    cache -- cache of Z and A computed in forward-propagation
    parameters -- dictionary of parameters W end b

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...     
    """

    m = Y.shape[1]          # number of examples
    L = len(cache) // 2     # number of layers (must // 2 because cache contains 'A' e 'Z')

    grads = {}

    # output layer
    l = L
    A = Yhat
    Z = cache["Z" + str(l)]          # get Z[l] from cache
    W = parameters["W" + str(l)]     # get W[l] from parameters

    #dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A)) # derivative of cost with respect to Yhat
    #dZ = dA * dsigmoid(Z) or else dZ = A - Y
    dZ = A - Y
    dW = (1 / m) * np.dot(dZ, cache["A" + str(l-1)].T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA = np.dot(W.T, dZ) 
    grads["dW" + str(l)] = dW
    grads["db" + str(l)] = db
    grads["dA" + str(l - 1)] = dA

    # inner layers
    for l in reversed(range(1, L)):
        Z = cache["Z" + str(l)]          # get Z[l] from cache
        W = parameters["W" + str(l)]     # get W[l] from parameters
        dZ = dA * drelu(Z)
        dW = (1 / m) * np.dot(dZ, cache["A" + str(l-1)].T)
        db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        dA = np.dot(W.T, dZ) 

        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
        grads["dA" + str(l - 1)] = dA

    return grads
    
def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2    # number of layers (must // 2 because parameters contains 'W' e 'b')

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)] 
    
    return parameters

def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []          # keep track of cost
    accuracies = []     # keep track of accuracy
    
    parameters = initialize_parameters(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        Yhat, cache = forwardpropagation(X, parameters)
        
        # Compute cost.
        cost = compute_cost(Y, Yhat)
    
        # Backward propagation.
        grads = backpropagation(Y, Yhat, cache, parameters)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            pred = predict(X, parameters)   # Current prediction
            accuracy = np.mean(pred == Y)   # Compute accuracy
            print ("Cost after iteration %i: %f - accuracy: %f" %(i, cost, accuracy))

        if print_cost and i % 100 == 0:
            costs.append(cost)
            accuracies.append(accuracy)
            
    # plot the cost
    plt.plot(np.squeeze(costs), label="cost")
    plt.plot(np.squeeze(accuracies), label="accuracy")
    #plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def predict(X, parameters):

    m = X.shape[1]
    probs, cache = forwardpropagation(X, parameters)

    pred = np.array(probs, copy = True)
    pred[probs > .5] = 1
    pred[probs <= .5] = 0
    return pred
