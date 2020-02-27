import argparse

import os
import time
import random
import uuid

import json

import numpy as np
from numpy import dtype, shape

import tensorflow as tf
import helper_functions as hf 
import feature_evaluation as eval

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import pickle

from scipy.spatial import distance
from _operator import length_hint

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.hf.read_data_sets("MNIST_data/", one_hot=True)

# Hyperparameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100

dataset_name = 'mnist' #mnist'
input_json_dir = 'nnet_archs/'
input_json = 'mnist_net.json'

# Set random seed for replication
seed=1234
hf.set_seed(seed=seed)
display_step = 1
evaluate_features_flag = False

# Tuning parameters
tuning_type = 'centrality' # tuning type:Centrality-based (default) or KL Divergence
scale_type = 'minmax'
shift_type = 'min' # type of shifting the target distribution to positive values in KL div-based tuning
target_distribution = 'norm' # target distribution in KL div-based tuning
tuning_step = 1 # number of step(s) at which centrality-based tuning periodically takes place
k_selected = 2 # number of neurons selected to be tuned ('turned off') each round
n_tuned_layers = 1 # number of layers to be tuned; a value of 2 means layers 2 and 3 (1st & 2nd hidden layers will be tuned)
percentiles=False
noise_ratio = 0.0

output_dir = "results/"
uid = str(uuid.uuid4())[0:8] # unique id
print('UID: ' +str(uid))

# Parse input arguments
# sample command with tuning: python hf.tune_weights.py --tune 1
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument("--ts", type=float, help="Tuning step size")
parser.add_argument("--tt", help="Tuning type")
parser.add_argument("--sd", type=int, help="Randomization seed")
parser.add_argument("--nt", type=int, help="Number of tuned layers")
parser.add_argument("--ds", help="Dataset name")
parser.add_argument("--ij", help="Input network JSON file")
parser.add_argument("--st", help="Distribution shift type") # shift type of the weight distribution to positive
parser.add_argument("--td", help="Target distirubion for KL-divergence based comparisons.") # shift type of the weight distribution to positive
parser.add_argument("--nr", type=float, help="Noise ratio (added to data)")

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
    hf.set_seed(seed=seed) # set same seed in helper
    print("Randomization seed set to {}".format(seed))
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

if args.nr:
    noise_ratio = args.nr
    print("Noise ratio set to {}".format(noise_ratio))
         
with open(input_json_dir+input_json) as json_file:    
    json_data = json.load(json_file)

    layer_sizes = json_data['layers']['sizes'] # In, h1, h2, ..., Out
    n_input = layer_sizes[0]
    n_classes = layer_sizes[-1]
    
    layer_types = json_data['layers']['types'] # In, h1, h2, ..., Out
    activ_funcs = json_data['layers']['activ_funcs'] # h1, h2, ..., Out

    training_epochs = json_data['training_params']['epochs']
    learning_rate = json_data['training_params']['learning_rate']
    batch_size = json_data['training_params']['batch_size']
    
    eval_type = json_data['evaluation']['eval_type']
    top_k = json_data['evaluation']['top_k']
    bottom_features = json_data['evaluation']['bottom_features']
    visualize_imgs = json_data['evaluation']['visualize_imgs']
    n_imgs = json_data['evaluation']['n_imgs']
    sorted_ref_features = json_data['evaluation']['sorted_ref_features']
    discarded_features = json_data['evaluation']['discarded_features']

    print("Evaluation type: {0}\nTop k: {1}\nBottom Features Flag: {2}\nVisualize Images: {3}\nN_imgs: {4}\nSorted_ref_features: {5}\nDiscarded_features: {6}".format(eval_type, top_k, bottom_features, visualize_imgs, n_imgs, sorted_ref_features, discarded_features))
    print('Layer sizes: {0} \n Layer types: {1} \n Activation functions: {2} \n Epochs: {3} \n Learning rate: {4} \n Batch size: {5}'.format(layer_sizes, layer_types, activ_funcs, training_epochs, learning_rate, batch_size))

# Complement available indices above. Updated at each neuron tuning step.
off_indices = hf.get_arrdict(layer_sizes, 'empty', 'o')
avail_indices = hf.get_arrdict(layer_sizes, 'range', 'a')

# Input data placeholder
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
    
# Store layers weight & bias
weight_init = 'norm'
weight_init_reduction='fan_in'
bias_init = 'norm'

weights = hf.get_vardict(layer_sizes, weight_init, 'weight', 'w', seed=seed)
tuned_weights = hf.get_vardict(layer_sizes, 'zeros', 'weight', 'w',  weight_init_reduction= weight_init_reduction, seed=seed)
biases = hf.get_vardict(layer_sizes, bias_init, 'bias', 'b', seed=seed)

# Neuron tuning tf operation
neuron_tuning_op_w1 = tf.assign(weights['w1'], tuned_weights['w1'])
neuron_tuning_op_w2 = tf.assign(weights['w2'], tuned_weights['w2'])
neuron_tuning_op_w3 = tf.assign(weights['w3'], tuned_weights['w3'])

# Construct the model
logits = hf.multilayer_perceptron(X, weights, biases, activ_funcs, layer_types)

# Define loss function and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables
init = tf.global_variables_initializer()

D = hf.read_dataset(dataset_name, noise_ratio=noise_ratio, seed=seed)
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
        if(int(epoch % tuning_step) == 0 and (args.ts or args.tt)):
            # choose layers on which tuning is executed
            tuning_layer_start = 2
            tuning_layer_end = tuning_layer_start+n_tuned_layers
            
            if tuning_layer_end > len(weights_dict)+1:
                tuning_layer_end = len(weights_dict)+1
                print("Tuning layer end is out of bounds. Set to {}".format(tuning_layer_end))
            
            weights_dict = sess.run(weights)    
            for l in range(2, tuning_layer_end):                
                print("Tuning on layer {}".format(l))        

                current_off_indices = off_indices['o'+str(l)]
                #print(current_off_indices)
                new_off_inds = hf.get_off_inds(weights_dict, avail_inds=avail_indices['a'+str(l)], off_inds=current_off_indices, 
                                            layer_index=l, k_selected=k_selected, tuning_type=tuning_type, shift_type=shift_type, 
                                            scale_type=scale_type, target_distribution=target_distribution, percentiles=percentiles)

                # update available and off_indices (i.e. indices of tuned neurons)
                avail_indices['a'+str(l)] = np.delete(avail_indices['a'+str(l)], np.searchsorted(avail_indices['a'+str(l)], new_off_inds))
                off_indices['o'+str(l)] = np.append(off_indices['o'+str(l)], new_off_inds)

                # get a tensor with off_inds neurons turned off
                tuned_weights['w'+str(l)] = hf.tune_weights(off_indices['o'+str(l)], weights_dict, l)
                print("Weight tuning done.")

            # run neuron tuning operation(s)
            sess.run([neuron_tuning_op_w1, neuron_tuning_op_w2]) # first and second matrices surrounding first layer
            if(n_tuned_layers == 2): # if second layer needs to be tuned
                sess.run(neuron_tuning_op_w3)

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

    pred = tf.nn.softmax(logits)  # Apply softmax to logits

    # Test model
    ts_predictions = sess.run(pred, feed_dict={X: X_ts, Y: Y_ts})
    accuracy = accuracy_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
    print("Accuracy:", accuracy)

    if(n_classes == 2):
        precision = precision_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
        print("Precision:", precision)
        
        auc = roc_auc_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
        print("AUC ROC:", auc)

    if evaluate_features_flag:
        weights_dict = sess.run(weights)
        weights_filename = output_dir+str(uid)+'_weights.pkl'
        pickle.dump(weights_dict, open(weights_filename, 'wb'))
        print("Pickled weights in " +weights_filename)
    
        bias_dict = sess.run(biases)
        biases_file = open(output_dir+str(uid)+'_biases.pkl', 'wb')
        pickle.dump(bias_dict, biases_file)
        print("Pickled bias values.")

# Feature evaluation
if evaluate_features_flag:
    weights_dict = pickle.load(open(weights_filename, 'rb'))
    
    tuning_measure = ''
    if 'kl_div' in tuning_type or 'ks_test' in tuning_type:
        tuning_measure = (tuning_type +":"+ target_distribution)
    
    scoring_functions = [tuning_measure, 'min', 'max', 'avg', 'median', 'skew', 'kurt', 'std', 'abs_'+tuning_measure, 'abs_min', 'abs_max', 'abs_avg', 'abs_median', 'abs_skew', 'abs_kurt', 'abs_std']
    eval.evaluate_features(dataset_name=dataset_name, weights_dict=weights_dict, scoring_functions=scoring_functions, eval_type=eval_type, sorted_ref_features=sorted_ref_features, discarded_features=discarded_features, output_dir='results/', uid=uid, top_k=top_k, input_data=X_ts, bottom_features=bottom_features, visualize_imgs=visualize_imgs, n_imgs=n_imgs)