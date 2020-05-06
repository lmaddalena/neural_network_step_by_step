import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from neuralnetwork import *
from utils import *


def main():
    print("Deep Neural Network Step-by-Step")
    print("Rif.: Neural Networks and Deep Learning - week 4 (revisited)")

    print("")
    print("Loading dataset...\n")
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

    # Explore your dataset 
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))
    print ("-----------------------------------------------")

    # Example of a picture
    index = 10
    print("Image idx: " + str(index))
    #plt.imshow(train_x_orig[index])
    #plt.show()
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    print ("-----------------------------------------------")

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    print("")

    layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

    parameters = model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    print("\nPrediction on test dataset")
    pred = predict(test_x, parameters)
    accuracy = np.mean(pred == test_y)
    print ("Accuracy on test dataset: %f\n" %(accuracy))


if __name__ == "__main__":
    main()