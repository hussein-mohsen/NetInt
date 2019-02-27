from __future__ import print_function

import argparse

import os
import time
import random

import numpy as np
from numpy import dtype, shape

import tensorflow as tf

from helper_functions import get_layer_inds, get_off_inds, pad_matrix, tune_weights, create_weight_graph
# import MNIST_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# Hyperparameters
learning_rate = 0.001
training_epochs = 300
batch_size = 100

n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 512
n_hidden_2 = 512
n_classes = 10 # MNIST total classes (0-9 digits)

# Set random seed for replication
seed=1234
tf.set_random_seed(seed)
np.random.seed(seed=seed)
random.seed(seed)
display_step = 2

# Indices of available neurons in input and hidden layers.
# Updated whenever centrality-based tuning takes place by eliminating neurons 'turned off'
avail_indices = {
    'a1': np.array(range(n_input)), # available neuron indices in input layer
    'a2': np.array(range(n_hidden_1)), # available neuron indices in hidden layer 1
    'a3': np.array(range(n_hidden_2)) # available neuron indices in hidden layer 2
}

# Complement available indices above. Updated at each neuron tuning step.
off_indices = {
    'o1': np.array([], dtype=int), # off indices in input layer
    'o2': np.array([], dtype=int), # off indices in hidden layer 1
    'o3': np.array([], dtype=int) # off indices in hidden layer 2
}

tuning_step = 1 # number of step(s) at which centrality-based tuning periodically takes place
k_selected=1 # number of neurons selected to be tuned ('turned off') each round
n_tuned_layers = 1 # number of layers to be tuned; a value of 2 means layers 2 and 3 (1st & 2nd hidden layers will be tuned)

# Input data placeholder
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

# tensors with tuned weights (weights above with incoming 
# and outcoming weights of tuned neurons set to 0 values)
tuned_weights = {
    'w1': tf.Variable(tf.zeros([n_input, n_hidden_1], dtype=tf.float32)),
    'w2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2], dtype=tf.float32)),
    'w3': tf.Variable(tf.zeros([n_hidden_2, n_classes], dtype=tf.float32))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_classes]))
}

# Create the MLP 
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return out_layer

# Construct the model
logits = multilayer_perceptron(X)

# Neuron tuning tf operation
neuron_tuning_op = tf.assign(weights['w2'], tuned_weights['w2'])

# Define loss function and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables
init = tf.global_variables_initializer()

# Parse input arguments
# sample command with tuning: python tune_weights.py --tune 1
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument("--ts", type=int, help="Tuning step size")
parser.add_argument("--sd", type=int, help="Randomization seed")
parser.add_argument("--ep", type=int, help="Number of epochs")
parser.add_argument("--nt", type=int, help="Number of tuned layers")
args = parser.parse_args()

if args.ts:
    tuning_step = args.ts
    print("Centrality measure tuning step set to {}".format(tuning_step))
    
if args.sd:
    seed = args.sd
    print("Randomization seed set to {}".format(seed))

if args.ep:
    training_epochs = args.ep
    print("Epochs set to {}".format(training_epochs))

if args.nt:
    n_tuned_layers = args.nt
    print("Number of tuned layers set to {}".format(n_tuned_layers))

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(1, training_epochs+1):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        start = time.time()

        # centrality-based neuron tuning step
        if(epoch % tuning_step == 0 and args.ts):
            weights_dict = sess.run(weights)
            
            # choose layers on which tuning is executed
            tuning_layer_start = 2
            tuning_layer_end = tuning_layer_start+n_tuned_layers
            
            if tuning_layer_end > len(weights_dict)+1:
                tuning_layer_end = len(weights_dict)+1
                print("Tuning layer end is out of bounds. Set to {}".format(tuning_layer_end))
                
            for l in range(2, tuning_layer_end):
                print("Tuning on layer {}".format(l))        
                # create weight graph
                weight_graph, layer_boundaries = create_weight_graph(weights_dict, l)
                current_off_inds = get_off_inds(weight_graph, avail_indices['a'+str(l)], layer_boundaries=layer_boundaries, k_selected=k_selected, centrality='btw', random=False)
                
                # update available and off_indices (i.e. indices of tuned neurons)
                avail_indices['a'+str(l)] = np.delete(avail_indices['a'+str(l)], current_off_inds)
                off_indices['o'+str(l)] = np.append(off_indices['o'+str(l)], current_off_inds)

                # get a tensor with off_inds neurons turned off
                tuned_weights['w'+str(l)] = tune_weights(off_indices['o'+str(l)], weights_dict, l)
                print("Weight tuning done.")

            # run neuron tunin operation
            sess.run(neuron_tuning_op)
            
         # Loop over all batches
        for i in range(total_batch):        
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)

            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            
            # Compute average loss
            avg_cost += c / total_batch
            
        end = time.time()
        print('Execution Time: {0} {1}'.format(1000*(end-start), 'ms'))
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            
    print("Optimization Done.")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))