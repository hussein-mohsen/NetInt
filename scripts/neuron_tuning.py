from __future__ import print_function

import argparse

import os
import time
import random

import numpy as np
from numpy import dtype, shape

import tensorflow as tf

from helper_functions import get_layer_inds, get_off_inds, pad_matrix, tune_weights, create_weight_graph, read_dataset, get_next_batch, set_seed, get_layer

# Hyperparameters
learning_rate = 0.001
training_epochs = 50 #300 #50
batch_size = 100
dataset_name = 'mnist'

n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 512
n_hidden_2 = 512
n_classes = 10 # MNIST total classes (0-9 digits)

#n_input = 932 # MNIST data input (img shape: 28*28)
#n_hidden_1 = 400
#n_hidden_2 = 100
#n_classes = 2 # MNIST total classes (0-9 digits)

# Set random seed for replication
seed=1234
tf.set_random_seed(seed)
np.random.seed(seed=seed)
random.seed(seed)
display_step = 1

# Tuning parameters
tuning_type = 'centrality' # tuning type:Centrality-based (default) or KL Divergence
tuning_step = 1 # number of step(s) at which centrality-based tuning periodically takes place
k_selected=1 # number of neurons selected to be tuned ('turned off') each round
n_tuned_layers = 1 # number of layers to be tuned; a value of 2 means layers 2 and 3 (1st & 2nd hidden layers will be tuned)

# Parse input arguments
# sample command with tuning: python tune_weights.py --tune 1
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument("--ts", type=int, help="Tuning step size")
parser.add_argument("--tt", help="Tuning type")
parser.add_argument("--sd", type=int, help="Randomization seed")
parser.add_argument("--ep", type=int, help="Number of epochs")
parser.add_argument("--nt", type=int, help="Number of tuned layers")
parser.add_argument("--ds", help="Dataset name")

args = parser.parse_args()

if args.ts:
    tuning_step = args.ts
    print("Tuning step set to {}".format(tuning_step))

if args.tt:
    tuning_type = args.tt
    print("Tuning type set to {}".format(n_tuned_layers))
    
if args.sd:
    seed = args.sd
    set_seed(seed) # set same seed in helper
    print("Randomization seed set to {}".format(seed))

if args.ep:
    training_epochs = args.ep
    print("Epochs set to {}".format(training_epochs))

if args.nt:
    n_tuned_layers = args.nt
    print("Number of tuned layers set to {}".format(n_tuned_layers))

if args.ds:
    dataset_name = args.ds
    print("Dataset set to {}".format(dataset_name))
    
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

activ_funcs = ['linear', 'linear', 'linear']

# Create the MLP
# x is the input tensor, activ_funs is a list of activation functions
# weights and biases are dictionaries with keys = 'w1'/'b1', 'w2'/'b2', etc.
def multilayer_perceptron(x, weights, biases, activ_funcs):
    layers = []
    
    for l in range(len(activ_funcs)):
        if(l == 0):
            input_tensor = x
        else:
            input_tensor = layers[l-1]
            
        weights_key = 'w' + str(l+1)
        biases_key = 'b' + str(l+1)
        activation_function = activ_funcs[l]
        
        layer = get_layer(input_tensor, weights[weights_key], biases[biases_key], activation_function)
        layers.append(layer)

    return layers[-1] # return the last tensor, i.e. output layer

# Neuron tuning tf operation
neuron_tuning_op2 = tf.assign(weights['w2'], tuned_weights['w2'])
neuron_tuning_op3 = tf.assign(weights['w3'], tuned_weights['w3'])

# Construct the model
logits = multilayer_perceptron(X, weights, biases, activ_funcs)

# Define loss function and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables
init = tf.global_variables_initializer()

# load data
X_tr, Y_tr, X_ts, Y_ts = read_dataset(dataset_name)

with tf.Session() as sess:
    sess.run(init)

    print("Training has started.")
    
    if(X_tr.shape[0] < 1000):
        batch_size = X_tr.shape[0]
        
    total_batch = int(X_tr.shape[0]/batch_size) + (X_tr.shape[0]%batch_size != 0)
    
    # Training cycle
    for epoch in range(1, training_epochs+1):
        avg_cost = 0.0
        
        start = time.time()
        print("\nEpoch:", '%04d' % (epoch))

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
                current_off_inds = get_off_inds(weights_dict, avail_indices['a'+str(l)], layer_index=l, k_selected=k_selected, tuning_type=tuning_type)
                
                # update available and off_indices (i.e. indices of tuned neurons)
                avail_indices['a'+str(l)] = np.delete(avail_indices['a'+str(l)], current_off_inds)
                off_indices['o'+str(l)] = np.append(off_indices['o'+str(l)], current_off_inds)

                # get a tensor with off_inds neurons turned off
                tuned_weights['w'+str(l)] = tune_weights(off_indices['o'+str(l)], weights_dict, l)
                print("Weight tuning done.")
            
            # run neuron tuning operation
            if(n_tuned_layers == 1):
                sess.run(neuron_tuning_op2)
            elif(n_tuned_layers == 2):
                sess.run([neuron_tuning_op2, neuron_tuning_op3])
                
         # Loop over all batches
        for i in range(total_batch):            
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = get_next_batch(X_tr, Y_tr, i, batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            
            # Compute average loss
            avg_cost += c / total_batch
            
        end = time.time()
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print('Execution Time: {0} {1}, Cost: {2}'.format(1000*(end-start), 'ms', avg_cost))
            
    print("Optimization Done.")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: X_ts, Y: Y_ts}))