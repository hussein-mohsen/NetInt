import argparse

import os
import time
import random

import json

import numpy as np
from numpy import dtype, shape

import tensorflow as tf

from helper_functions import get_layer_inds, get_off_inds, pad_matrix, tune_weights, create_weight_graph, read_dataset, get_next_batch, get_next_even_batch, set_seed, get_layer, get_vardict, get_arrdict, multilayer_perceptron, ks_test
from numpy.random.mtrand import shuffle

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hyperparameters
learning_rate = 0.001
training_epochs = 300
batch_size = 100

dataset_name = 'mnist'
input_json_dir = 'nnet_archs/'
input_json = 'mnist_net.json'

# Set random seed for replication
seed=1234
tf.set_random_seed(seed)
np.random.seed(seed=seed)
random.seed(seed)
display_step = 1

# Tuning parameters
tuning_type = 'centrality' # tuning type:Centrality-based (default) or KL Divergence
shift_type = 'min' # type of shifting the target distribution to positive values in KL div-based tuning
target_distribution = 'norm' # target distribution in KL div-based tuning
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
parser.add_argument("--ij", help="Input network JSON file")
parser.add_argument("--st", help="Distribution shift type") # shift type of the weight distribution to positive
parser.add_argument("--td", help="Target distirubion for KL-divergence based comparisons.") # shift type of the weight distribution to positive

args = parser.parse_args()

if args.ts:
    tuning_step = args.ts
    display_step = tuning_step
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
    input_json = dataset_name+'_net.json'
    print("Dataset set to {}".format(dataset_name))
    print("Input JSON file set to {}".format(input_json))

if args.ij:
    input_json = args.ij
    print("Input JSON file set to {}".format(input_json))

if args.st:
    shift_type = args.st
    print("Distribution shift type set to {}".format(shift_type))

if args.td:
    target_distribution = args.td
    print("Target distribution set to {}".format(target_distribution))
        
with open(input_json_dir+input_json) as json_file:    
    json_data = json.load(json_file)

    layer_sizes = json_data['layers']['sizes'] # In, h1, h2, ..., Out
    n_input = layer_sizes[0]
    n_classes = layer_sizes[-1]
    
    layer_types = json_data['layers']['types'] # In, h1, h2, ..., Out
    activ_funcs = json_data['layers']['activ_funcs'] # h1, h2, ..., Out

# Complement available indices above. Updated at each neuron tuning step.
off_indices = get_arrdict(layer_sizes, 'empty', 'o')
avail_indices = get_arrdict(layer_sizes, 'range', 'a')

# Input data placeholder
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
    
# Store layers weight & bias
weight_init = 'norm'
bias_init = 'norm'

weights = get_vardict(layer_sizes, weight_init, 'weight', 'w')
tuned_weights = get_vardict(layer_sizes, 'zeros', 'weight', 'w')
biases = get_vardict(layer_sizes, bias_init, 'bias', 'b')

# Neuron tuning tf operation
neuron_tuning_op2 = tf.assign(weights['w2'], tuned_weights['w2'])
neuron_tuning_op3 = tf.assign(weights['w3'], tuned_weights['w3'])

# Construct the model
logits = multilayer_perceptron(X, weights, biases, activ_funcs, layer_types)

# Define loss function and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables
init = tf.global_variables_initializer()

D = read_dataset(dataset_name)
X_tr, Y_tr = D.train.points, D.train.labels
X_ts, Y_ts = D.test.points, D.test.labels

with tf.Session() as sess:
    sess.run(init)

    print("Training has started.")
    
    if(X_tr.shape[0] < 1000):
        batch_size = X_tr.shape[0]
        print("Batch size set to {}".format(batch_size))
        
    total_batch = int(X_tr.shape[0]/batch_size) # + (X_tr.shape[0]%batch_size != 0)
    
    # Training cycle
    for epoch in range(1, training_epochs+1):
        avg_cost = 0.0
        
        start = time.time()
        
        # centrality-based neuron tuning step
        if(epoch % tuning_step == 0 and (args.ts or args.tt)):
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
                current_off_inds = get_off_inds(weights_dict, avail_indices['a'+str(l)], layer_index=l, k_selected=k_selected, 
                                                tuning_type=tuning_type, shift_type=shift_type, target_distribution=target_distribution,
                                                percentiles=True)
                
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
            batch_x, batch_y = D.train.next_batch(batch_size)

            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            
            # Compute average loss
            avg_cost += c / total_batch
            
        end = time.time()
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("\nEpoch:", '%04d' % (epoch))
            print('Execution Time: {0} {1}, Cost: {2}'.format(1000*(end-start), 'ms', avg_cost))
            
    print("Optimization Done.")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: X_ts, Y: Y_ts}))