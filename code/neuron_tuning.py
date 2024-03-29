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
from scipy.special import expit

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.hf.read_data_sets("MNIST_data/", one_hot=True)

# Hyperparameters
learning_rate = 0.001
regularization='L2'
regularization_beta=0.001
training_epochs = 100
batch_size = 100

dataset_name = 'mnist' #mnist'
input_json_dir = 'nnet_archs/'
input_json = 'mnist-nonlinear_net.json'

# Set random seed for replication
seed=np.random.randint(9999)
hf.set_seed(seed=seed)
evaluate_features_flag = False
save_weights = True

# Tuning parameters
tuning_type = 'kl_div' # tuning type: Centrality-based (default) or KL Divergence
shift_type = 'min' # type of shifting the target distribution to positive values in KL div-based tuning
target_distribution = 'norm' # target distribution in KL div-based tuning
tuning_step = 2 # number of step(s) at which centrality-based tuning periodically takes place
tuning_layer_start = 2
            
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
parser.add_argument("--td", help="Target distirubion for KL-divergence based comparisons") # shift type of the weight distribution to positive
parser.add_argument("--nr", type=float, help="Noise ratio (added to data)")
parser.add_argument("--ar", type=int, help="Architecture reduction boolean")
parser.add_argument("--rg", help="Regularization technique") # 'none' or 'L2'

args = parser.parse_args()

if args.ts:
    tuning_step = args.ts
    print("Tuning step set to {}".format(tuning_step))

if args.tt:
    tuning_type = args.tt
    print("Tuning type set to {}".format(tuning_type))
    
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
    
if args.rg:
    regularization = args.rg
    print("Regularization technique set to {}".format(regularization))
         
tuning_flag = False
if args.ts or args.tt:
    tuning_flag = True
    print("Tuning flag turned on")

arch_reduction_flag = False
if args.ar == 1:
    arch_reduction_flag = True
    print("Architecture reduction flag turned on")
    
with open(input_json_dir+input_json) as json_file:    
    json_data = json.load(json_file)

    layer_sizes = json_data['layers']['sizes'] # In, h1, h2, ..., Out
    n_input = layer_sizes[0]
    n_classes = layer_sizes[-1]
    
    layer_types = json_data['layers']['types'] # layer_types indexing: layer_types[0] > layer 1 (input layer), layer_types[1] > layer 2 (1st hidden layer), layer_types[-1] > output layer
    activ_funcs = json_data['layers']['activ_funcs'] # activation function indexing: activ_funcs[0] > layer 2, activ_funcs[1] > layer 3, etc.

    training_epochs = json_data['training_params']['epochs']
    learning_rate = json_data['training_params']['learning_rate']
    batch_size = json_data['training_params']['batch_size']
    regularization_beta = json_data['training_params']['regularization_beta']
    
    eval_type = json_data['evaluation']['eval_type']
    top_k = json_data['evaluation']['top_k']
    bottom_features = json_data['evaluation']['bottom_features']
    visualize_imgs = json_data['evaluation']['visualize_imgs']
    n_imgs = json_data['evaluation']['n_imgs']
    sorted_ref_features = json_data['evaluation']['sorted_ref_features']
    discarded_features = json_data['evaluation']['discarded_features']

    print("Evaluation type: {0}\nTop k: {1}\nBottom Features Flag: {2}\nVisualize Images: {3}\nN_imgs: {4}\nSorted_ref_features: {5}\nDiscarded_features: {6}".format(eval_type, top_k, bottom_features, visualize_imgs, n_imgs, sorted_ref_features, discarded_features))
    print('Layer sizes: {0} \n Layer types: {1} \n Activation functions: {2} \n Epochs: {3} \n Learning rate: {4} \n Batch size: {5} \n Seed: {6}\n Regularization: {7}\n Regularization Beta: {8}'.format(layer_sizes, layer_types, activ_funcs, training_epochs, learning_rate, batch_size, seed, regularization, regularization_beta))

if tuning_flag:
    tuning_layer_end = tuning_layer_start + n_tuned_layers - 1 # choose layers on which tuning is executed
    print('Tuning layer start, end: {0}, {1}'.format(tuning_layer_start, tuning_layer_end))

if arch_reduction_flag:
    layer_sizes = hf.reduce_architecture(layer_sizes, tuning_step, training_epochs, k_selected, n_tuned_layers, start_layer=1) # start layer 1 corresponds to layer 2, i.e. first hidden layer
    tuning_flag = False
    print("Architecture reduction done. Tuning flag turned off.")

display_step = min(100, int(training_epochs/2))

# Network and training setup
# Complement available indices above. Updated at each neuron tuning step.
off_indices = hf.get_arrdict(layer_sizes, 'empty', 'o')
avail_indices = hf.get_arrdict(layer_sizes, 'range', 'a')

# Input data placeholder
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# read dataset
D = hf.read_dataset(dataset_name, noise_ratio=noise_ratio, seed=seed)
X_tr, Y_tr = D.train.points, D.train.labels
X_val, Y_val = D.validation.points, D.validation.labels

X_tr = np.vstack((X_tr, X_val))
Y_tr = np.vstack((Y_tr, Y_val))

X_ts, Y_ts = D.test.points, D.test.labels

# store layers weight & bias dict
weight_init = 'norm'
bias_init = 'norm'

init_reduction='fan_in'
fanin_std_devs = []
if init_reduction == 'fan_in':
    fanin_std_devs = hf.get_fanin_stds(X_tr, layer_sizes, activ_funcs, selected_range=(-3, 3))

weights = hf.get_vardict(layer_sizes, weight_init, 'weight', 'w', init_reduction, fanin_std_devs, seed=seed)
tuned_weights = hf.get_vardict(layer_sizes, 'zeros', 'weight', 'w', seed=seed)
biases = hf.get_vardict(layer_sizes, bias_init, 'bias', 'b', init_reduction, fanin_std_devs, seed=seed)

# Construct the model
nnet = hf.multilayer_perceptron(X, weights, biases, activ_funcs, layer_types)

# Define loss function and optimizer
output_layer_index = len(layer_sizes)-1 # since indexing starts from 0
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nnet[output_layer_index], labels=Y))

# regularization
if regularization == 'L2':
    if 'TCGA_survival' in dataset_name:
        regularizers = tf.cast(tf.nn.l2_loss(weights['w1']), tf.float64)
    else:
        regularizers = tf.nn.l2_loss(weights['w1'])
    
    for i in range(2, (len(weights)+1)):
        if 'TCGA_survival' in dataset_name:
            regularizers = regularizers + tf.cast(tf.nn.l2_loss(weights['w' +str(i)]), tf.float64)
        else:
            regularizers = regularizers + tf.nn.l2_loss(weights['w' +str(i)])
            
    print('L2 Regularization added.')
    loss_op = tf.reduce_mean(loss_op + (regularization_beta * regularizers))  
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    print("Training has started.")
    
    if(X_tr.shape[0] < 1000):
        batch_size = X_tr.shape[0]
        print("Batch size set to {}".format(batch_size))
        
    total_batch = int(X_tr.shape[0]/batch_size) # + (X_tr.shape[0]%batch_size != 0)
    print('Number of batches: {0}'.format(total_batch))
    
    # Training cycle
    for epoch in range(1, training_epochs+1):
        avg_cost = 0.0
        start = time.time()

        # tuning step
        if(int(epoch % tuning_step) == 0 and tuning_flag):
            nnet_values = sess.run(nnet, feed_dict={X: X_val, Y: Y_val}) # nnet_values[l-1] corresponds to batch inputs (wa+b's) to layer l; shape: (batch_size, number of destination neurons in layer l)

            for l in range(tuning_layer_start, tuning_layer_end+1):                
                print("Tuning on layer {}".format(l))
                current_off_indices = off_indices['o'+str(l)]        
                new_off_inds = hf.get_off_inds(nnet_values[str(l-1)+'i'], avail_inds=avail_indices['a'+str(l)], off_inds=current_off_indices,
                                               layer_index=l, k_selected=k_selected, tuning_type=tuning_type, target_distribution=target_distribution, 
                                               percentiles=percentiles)

                print(new_off_inds)
                
                # update available and off_indices (i.e. indices of tuned neurons)
                avail_indices['a'+str(l)] = np.delete(avail_indices['a'+str(l)], np.searchsorted(avail_indices['a'+str(l)], new_off_inds))
                off_indices['o'+str(l)] = np.append(off_indices['o'+str(l)], new_off_inds)

            # get tensors with off_inds neurons turned off
            hf.tune_weights(off_indices, weights, biases, tuning_layer_start, tuning_layer_end, sess=sess)
            hf.tune_masks(off_indices, nnet, tuning_layer_start, tuning_layer_end, sess=sess)
            print("Tuning done.")

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = D.train.next_batch(batch_size)

            '''
            ##r scrutinizing tensor values
            if i % 150 == 0: # scrutinize values on few batches per epoch only for convenience
                [weight_vals, logit_vals] = sess.run([weights, nnet], feed_dict={X: batch_x, Y:batch_y})
                for i in range(4):
                    if i > 0:
                        print('Min: {0:.2f}, Mean: {1:.2f}, Median: {2:.2f}, Max: {3:.2f} of w{4}'.format(np.min(weight_vals['w'+str(i)]), np.mean(weight_vals['w'+str(i)]), np.median(weight_vals['w'+str(i)]), np.max(weight_vals['w'+str(i)]), i))
                        print('Min: {0:.2f}, Mean: {1:.2f}, Median: {2:.2f}, Max: {3:.2f} of logits{4}i'.format(np.min(logit_vals[str(i)+'i']), np.mean(logit_vals[str(i)+'i']), np.median(logit_vals[str(i)+'i']), np.max(logit_vals[str(i)+'i']), i))
                        count = np.sum((logit_vals[str(i)+'i'] >= -3) & (logit_vals[str(i)+'i'] <= 3))
                        print(count/(logit_vals[str(i)+'i'].shape[0]*logit_vals[str(i)+'i'].shape[1]))            
            '''
            
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            
            # Compute average loss
            avg_cost += c / total_batch

        end = time.time()
            
        # Display logs per epoch step
        if epoch%1 == 0 or epoch == training_epochs or epoch % display_step == 0:
            print("\nEpoch:", '%04d' % (epoch))
            print('Execution Time: {0} {1}, Cost: {2}'.format(1000*(end-start), 'ms', avg_cost))
    
    
    # save weight and bias dictionaries
    if save_weights:
        pickle_basename = output_dir + dataset_name +'/'+ uid 
        [weight_vals, bias_vals] = sess.run([weights, biases], feed_dict={X: batch_x, Y:batch_y})
        pickle.dump(weight_vals, open(pickle_basename +'_weights.pkl', 'wb'))
        pickle.dump(weight_vals, open(pickle_basename +'_biases.pkl', 'wb'))
        print("Weight and bias tensors pickled under " +pickle_basename+ "*")
        
    print("Optimization Done.")

    pred = tf.nn.softmax(nnet[output_layer_index])  # Apply softmax to outputs

    # Test model
    tr_predictions = sess.run(pred, feed_dict={X: X_tr, Y: Y_tr})
    ts_predictions = sess.run(pred, feed_dict={X: X_ts, Y: Y_ts})
    
    tr_accuracy = accuracy_score(np.argmax(Y_tr, 1), np.argmax(tr_predictions, 1))
    ts_accuracy = accuracy_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
    print("Train Accuracy: {0}, Test Accuracy: {1}".format(tr_accuracy, ts_accuracy))

    if(n_classes == 2):
        tr_precision = precision_score(np.argmax(Y_tr, 1), np.argmax(tr_predictions, 1))
        ts_precision = precision_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
        print("Train Precision: {0}, Test Precision: {1}".format(tr_precision, ts_precision))
        
        tr_auc = roc_auc_score(np.argmax(Y_tr, 1), np.argmax(tr_predictions, 1))
        ts_auc = roc_auc_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
        print("Train AUC: {0}, Test AUC: {1}".format(tr_auc, ts_auc))

# Feature evaluation
if evaluate_features_flag:
    weights_npdict = pickle.load(open(weights_filename, 'rb'))
    
    tuning_measure = ''
    if 'kl_div' in tuning_type:
        tuning_measure = (tuning_type +":"+ target_distribution)
    
    scoring_functions = [tuning_measure, 'min', 'max', 'avg', 'median', 'skew', 'kurt', 'std', 'abs_'+tuning_measure, 'abs_min', 'abs_max', 'abs_avg', 'abs_median', 'abs_skew', 'abs_kurt', 'abs_std']
    eval.evaluate_features(dataset_name=dataset_name, weights_npdict=weights_npdict, scoring_functions=scoring_functions, eval_type=eval_type, sorted_ref_features=sorted_ref_features, discarded_features=discarded_features, output_dir='results/', uid=uid, top_k=top_k, input_data=X_ts, bottom_features=bottom_features, visualize_imgs=visualize_imgs, n_imgs=n_imgs)
