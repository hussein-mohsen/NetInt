from __future__ import print_function

import argparse

import os
import time
import random

import numpy as np
from numpy import dtype, shape

import tensorflow as tf

from sklearn.preprocessing import normalize
import networkx as nx

# import MNIST_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# Hyperparameters
learning_rate = 0.001
training_epochs = 220
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
display_step = 1

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


# helper function to get indices of layer at index in a graph
def get_layer_inds(boundaries, index):
    if(index < 1 or index > len(boundaries)):
        print("Index out of bounds.")
    elif(index == 1): # input layer
        start = 0
    else: # hidden layers
        start = boundaries[index-2]
    
    end = boundaries[index-1] -1
    
    return start, end

# gets indices to be tuned (i.e. turned off and replaced by 0 values)
# available indices are the ones not tuned prior to selection   
# If random = False: centrality-based tuning mode; else, random tuning mode        
def get_off_inds(weight_graph, avail_inds, layer_boundaries, input_list=[], k_selected=4, centrality='btw', random=False, dt=[('weight', float)]):            
    if(random == False): # sorted centrality-based selection
        weight_graph = weight_graph.astype(dt)
        weight_G = nx.from_numpy_matrix(weight_graph) # create graph object
        
        # calculate centrality measure values
        print("Calculating centrality measures...")
        cent_values = np.array(list(nx.betweenness_centrality(weight_G, k=7, weight='weight').values()))

        # select available indices
        cent_values = cent_values[avail_inds]
        
        # select k_selected nodes to tune
        cent_inds = np.argsort(cent_values)
        select_inds = cent_inds[0:k_selected]
    else: # random selection of indices
        select_inds = random.sample(range(len(avail_inds)), k_selected) # indices within avail_inds to be turned off
    
    off_inds = avail_inds[select_inds]
    return off_inds # return array of indices to be tuned

# helper function that pads a matrix to create an NxN graph used to
# create weight graphs on which centrality measures are calculated
def pad_matrix(input_matrix):
    rows_increment = max(0, input_matrix.shape[1]-input_matrix.shape[0])
    cols_increment = max(0, input_matrix.shape[0]-input_matrix.shape[1])

    input_matrix = np.pad(input_matrix, pad_width=((0, rows_increment), (0, cols_increment)), mode='constant', constant_values=(0, 0))
    return input_matrix

# helper function that turns off neurons at off_indices (0 value assignment)
# returns a tf tensor
def tune_weights(off_indices, current_weights, layer):
    weights_dict['w'+str(layer)][off_indices, :] = 0 # turn off connections outcoming from off_indices neurons in the layer
    print("Outcoming connections at layer {} tuned.".format(layer))
     
    if(layer > 1):
        weights_dict['w'+str(layer-1)][:, off_indices] = 0 # turn off connections incoming to off_indices neurons in the layer
        print("Incoming connections at layer {} tuned.".format(layer))
        
    return tf.convert_to_tensor(weights_dict['w'+str(layer)], dtype=tf.float32)

# creates a weight graph at an input layer
# Layer indexing starts at 1; numpy indexing starts at 0
def create_weight_graph(weights_dict, layer):
    if layer == 1: # input layer: edge case where weight graph is made of one layer
        return pad_matrix(weights_dict['w'+str(layer)])
    elif layer > len(weights_dict): # output layer or erroneous index
        raise Exception('Layer is out of bounds.')
    else: # hidden layer: preceding, current and next layer forming the graph 
        boundaries = []
        total_n = 0
        
        # create boundaries matrix to help slicing the weight graph
        for l in range(layer-1, layer+1):
                total_n += weights_dict['w'+str(l)].shape[0]
                boundaries.append(total_n)
        
        total_n += weights_dict['w'+str(l)].shape[1]
        boundaries.append(total_n)
            
        # create the graph of combined nodes implemented as an undirected graph
        # hence each weight matrix is added twice (as is and as transpose)
        weight_graph = np.zeros((total_n, total_n))
        for l in range(layer-1, layer+2):

            weight_matrix = weights_dict['w'+str(layer)]
            pre_weight_matrix = weights_dict['w'+str(layer-1)]
            
            layer_start = boundaries[0]
            layer_end = boundaries[1]
            matrix_end = total_n
                
            if(l == layer): # then two matrices' values to be added to weight_graph: transpose(wlayer-1) and wlayer                            
                #  w_layer: add incoming and outcoming weight matrices
                weight_graph[layer_start:layer_end, 0:layer_start] = np.transpose(pre_weight_matrix)
                weight_graph[layer_start:layer_end, layer_end:matrix_end] = weight_matrix
            elif(l<layer):
                # w_layer-1 (pre-layer): add incoming weights of w_layer
                weight_graph[0:layer_start, layer_start:layer_end] = pre_weight_matrix
            elif(l>layer):
                # w_layer+1 (post-layer): add outcoming weights from w_layer
                weight_graph[layer_end:matrix_end, layer_start:layer_end] = np.transpose(weight_matrix)

    return weight_graph, [layer_start, layer_end]

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
parser.add_argument("--tune", help="Tune centrality-based trimming")
parser.add_argument("--sd", type=int, help="Randomization seed")
args = parser.parse_args()

if args.tune:
    print("Centrality measure tuning is on.")
    
if args.sd:
    seed = args.sd
    print("Randomization seed: {}".format(args.sd))

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        start = time.time()

        # centrality-based neuron tuning step
        if(epoch % tuning_step == 0 and args.tune):
            weights_dict = sess.run(weights)
            
            for l in range(2, len(weights_dict)):        
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