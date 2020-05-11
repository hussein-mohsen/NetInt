import random
import datetime
import pickle

import numpy as np
from numpy import dtype, shape

import tensorflow as tf
import networkx as nx
import collections

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from numpy.random.mtrand import shuffle

from scipy.io import loadmat
from scipy.stats import powerlaw, norm, pearsonr

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

from scipy import stats
from scipy.spatial import distance
from scipy.stats import powerlaw, norm

from helper_objects import DataSet
from tensorflow.python.framework import dtypes

from hyperopt import space_eval
from setuptools.dist import Feature

import re
import os
import platform

epsilon = 0.00001

# scales values s.t. sum = 1
def totality_scale(values):
    total = values.sum() + epsilon
    return (values/total)

# two-sided vector clipping of values
# converts values below or above margin to (low or high) exremity values
def clip_vector(vector, selected_range, range_margin=0.05):   
    margin_length = (selected_range[1] - selected_range[0]) * range_margin
    lower_threshold = selected_range[0] + margin_length
    upper_threshold = selected_range[1] - margin_length
    vector[vector > upper_threshold] = selected_range[1]
    vector[vector < lower_threshold] = selected_range[0]

    return vector
   
# returns a pmf from of a vectors histogram
def calculate_histogram_pmf(vector, selected_range=(), n_bins=20):
    if len(selected_range) > 0: 
        vector = clip_vector(vector, selected_range, (1.0/n_bins))
        vector_histogram, bins = np.histogram(vector, bins=n_bins, range=selected_range)
    else:
        vector_histogram, bins = np.histogram(vector, bins=n_bins)
    
    return totality_scale(vector_histogram)

# calculates KL divergence of two distributions
def kl_div(empirical, target_dist):
    if(abs(empirical.sum()-1) > 0.05 or abs(target_dist.sum()-1) > 0.05):
        print('Warning: one or more distributions do not sum up to 1.')

    kl_div_value = (empirical * np.log((empirical + epsilon)/(target_dist + epsilon))).sum()
    return kl_div_value
    
# Calculate KL div between scaled weights (per row) with scaled target distribution
def calculate_kl_div_values(input_matrix, seed=1234,
                            target_distribution='norm', axis=1):

    n_samples = 10000
    if target_distribution == 'norm':
        target_dist = norm.rvs(loc=0, scale=0.3, size=n_samples, random_state=seed)
    elif target_distribution == 'inv_powerlaw': # power law and inverse power law distribution
        target_dist = (-1) * powerlaw.rvs(a=0.65, loc=-2, scale=4, size=n_samples, random_state=seed)
    else:
        raise Exception('Invalid target distribution.')

    # calculate histograms of empirical data and target_dist
    input_matrix = np.apply_along_axis(calculate_histogram_pmf, axis, input_matrix, (-2, 2))
    target_dist = calculate_histogram_pmf(target_dist)

    kl_values = np.apply_along_axis(kl_div, axis, input_matrix, target_dist)
    return kl_values


# calculate KS test to compare distance between CDFs of empirical and target_dist samples
def ks_test(empirical, target_dist, metric='D'):
    ks_results = stats.ks_2samp(empirical, target_dist)
    
    if metric == 'D':
        return ks_results[0]
    elif metric == 'p_value':
        return ks_results[1]

# Calculate KS distance (D or p-)value between scaled weights (per row) with scaled target distribution
def calculate_ks_values(input_matrix, seed=1234, target_distribution='norm', ks_metric='D',
                        selected_range=(-2, 2), range_margin=(1.0/20), axis=1):
    
    n_samples = 750
    if target_distribution == 'norm':
        target_dist = norm.rvs(loc=0, scale=0.3, size=n_samples, random_state=seed)
    elif target_distribution == 'inv_powerlaw':
        target_dist = (-1) * powerlaw.rvs(a=0.65, loc=-2, scale=4, size=n_samples, random_state=seed)
    else:
        raise Exception('Invalid target distribution.')
        
    input_matrix = np.apply_along_axis(clip_vector, axis, input_matrix, selected_range, range_margin)    
    ks_values = np.apply_along_axis(ks_test, axis, input_matrix, target_dist, ks_metric)
    return ks_values

# calculates distribution distance (of all rows) per tuning type
def calculate_distance_values(input_matrix, tuning_type='kl_div', target_distribution='norm', 
                              ks_metric='D',percentiles=False, axis=1):
    if(percentiles): # exclude outliers and keep values in [1st, 99th] percentiles
        input_matrix = percentile_input_matrix(input_matrix, 1, 99, axis=axis)
        
    if tuning_type == 'kl_div':
        distance_values = calculate_kl_div_values(input_matrix, target_distribution=target_distribution, 
                                                  axis=axis)
    elif tuning_type == 'ks_test':
        distance_values = calculate_ks_values(input_matrix, target_distribution=target_distribution, 
                                                ks_metric=ks_metric, axis=axis)

    return distance_values

# gets indices to be tuned (i.e. turned off and replaced by 0 values)
# available indices are the ones not tuned prior to selection   
# tuning_type: 'kl_div' (default: with Gaussian)   
#              'random': 
def get_off_inds(input_matrix, avail_inds, off_inds, layer_index, input_list=[], 
                 k_selected=4, tuning_type='kl_div', dt=[('weight', float)], 
                 target_distribution='norm', percentiles=False):
    if(len(avail_inds) == 0):
        select_inds = [-1]
        print('Warning: no more neurons to tune.')
    else:
        if tuning_type == 'random': # random selection of indices
            select_inds = random.sample(range(len(avail_inds)), k_selected) # indices within avail_inds to be turned off        
        else:
            ##? to implement betweenness centrality in this section as needed
            
            if tuning_type == 'kl_div' or tuning_type == 'ks_test':
                increasing_flag = False
                
                # calculate KL divergence from a target distribution (default: Gaussian)
                values = calculate_distance_values(input_matrix, tuning_type=tuning_type, 
                                                   target_distribution=target_distribution, percentiles=percentiles, axis=0)
                print('Calculating distribution distance values per {0}...'.format(tuning_type))
            else:
                raise Exception('Invalid tuning type value.')
    
            # select k_selected nodes to tune
            inds = np.argsort(values)
            if increasing_flag == False:
                inds = inds[::-1]
                
            inds = inds[~np.in1d(inds, off_inds)] # remove current off_inds
            select_inds = inds[0:k_selected]

    return select_inds # return array of indices to be tuned

# helper function that turns off neuron masks at off_indices (0 value assignment)
# masks are layers in nnet: nnet['1m'] corresponds to mask following layer 2, i. e. first hidden layer
def tune_masks(off_indices, nnet, tuning_layer_start, tuning_layer_end, sess):
    for i in range(tuning_layer_start, tuning_layer_end+1):
        zeros_replacement_tensor = tf.zeros(off_indices['o'+str(i)].size, tf.float32)
        updated = tf.scatter_update(nnet[str(i-1)+'m'], off_indices['o'+str(i)], zeros_replacement_tensor) # update tensor
        sess.run(tf.assign(nnet[str(i-1)+'m'], updated))
        print('Mask of layer {} tuned.'.format(i))
            
# helper function that turns off neuron weights at off_indices (0 value assignment)
# weights is a dictionary of tf.Variables; tuned layers are [tuned_layers_start, tuned_layers_end] 
# tuned weight matrices are wj s.t. j = [max(1, tuning_layer_start-1), tuned_layers_end] 
# e.g. by default, for layers 2 and 3, tuning layer start and end are 2 and 3, to-be-tuned matrices are w1, 2, and 3
# tuning directions = 'outgoing' (tunes off outgoing weights only) or 'outgoing_ingoing' (both directions)
def tune_weights(off_indices, weights, biases, tuning_layer_start, tuning_layer_end, sess, tuning_direction='outgoing_ingoing'):
    tuning_weight_start = max(1, tuning_layer_start-1)
    tuning_weight_end = tuning_layer_end

    # every weight matrix w_wi is affected by source and destination neuron, whose off_indices are at o_wi and o_wi+1
    # e.g. weights in w2 are tuned off based on off indices in source and destination neurons in layers 2 and 3 (o2, o3)
    for wi in range(tuning_weight_start, tuning_weight_end+1):
        tune_off_outgoing_weights(off_indices, weights, wi, sess)
        if 'ingoing' in tuning_direction:
            tune_off_ingoing_weights(off_indices, weights, biases, wi, sess)

# helper function that tunes off outgoing weights; entire rows in in weight matrix wi
# wi is the index of the weight matrix; weights is a dictionary of tf.Variables
def tune_off_outgoing_weights(off_indices, weights, wi, sess):
    if 'o'+str(wi) in off_indices and off_indices['o'+str(wi)].size > 0:
        off_indices_list = np.reshape(off_indices['o'+str(wi)], (off_indices['o'+str(wi)].size, 1)) # tf accepts np list of lists
        zeros_replacement_tensor = tf.zeros((off_indices['o'+str(wi)].size, weights['w'+str(wi)].shape[1]), tf.float32)
        updated = tf.scatter_nd_update(weights['w'+str(wi)], off_indices_list, zeros_replacement_tensor) # update tensor
        sess.run(tf.assign(weights['w'+str(wi)], updated))
        print('Outgoing weights from source layer in w{} tuned.'.format(wi))

# helper function that tunes off ingoing weights; entire columns in weight matrix wi
# wi is the index of the ingoing weight matrix and bias vector; weights and biases are tf.Variable dictionaries
# indices of biases are same as incoming weight matrices: e.g. 'b1' with 'w1' working with neurons of layer 2 (1st hidden layer)
def tune_off_ingoing_weights(off_indices, weights, biases, wi, sess):
    if 'o'+str(wi+1) in off_indices and off_indices['o'+str(wi+1)].size > 0:
        # Columns updated differently due to limitations in current TensorFlow
        off_indices_list = tf.constant(off_indices['o'+str(wi+1)], dtype=tf.int32)
        
        # weight tuning
        index_range = tf.range(weights['w'+str(wi)].shape[0], dtype=tf.int32) # all indices in source layer, used to generate pairs with target indices in destination
        mesh_grid = tf.meshgrid(index_range, off_indices_list, indexing='ij')
        target_index_pairs = tf.stack(mesh_grid, axis=2)
        target_index_pairs = tf.reshape(target_index_pairs, (target_index_pairs.shape[0]*target_index_pairs.shape[1], target_index_pairs.shape[2])) # reshaping them because I'm close to OCD
        zeros_replacement_tensor =  tf.zeros(target_index_pairs.shape[0], tf.float32)
        updated_weights = tf.scatter_nd_update(weights['w'+str(wi)], target_index_pairs, zeros_replacement_tensor)
        
        # bias tuning
        updated_biases = tf.scatter_update(biases['b'+str(wi)], off_indices['o'+str(wi+1)], tf.zeros([off_indices['o'+str(wi+1)].size]))
        
        sess.run([tf.assign(weights['w'+str(wi)], updated_weights), tf.assign(biases['b'+str(wi)], updated_biases)])
        print('Ingoing weights and biases into destination layer in w{0} and b{1} tuned.'.format(wi, wi))

# Creates the MLP
# x is the input tensor, activ_funs is a list of activation functions
# weights and biases are dictionaries with keys = 'w1'/'b1', 'w2'/'b2', etc.
# returns a dictionary of layers 0: input layer, '1i': wx+b input to first hidden layer, '1f': f(wx+b), '1m': 0 if neuron masked, 1 otherwise, '1': 1f*1m elementwise, '2i': same as 1i but for second hidden layer, ... 'k' for output layer
def multilayer_perceptron(x, weights, biases, activ_funcs, layer_types):
    layers_dict = {0: x} # layer indexing: layers_dict[0] is layer 1 (input), layers_dict[1] is layer 2 (1st hidden layer), layers_dict[last_layer_index] is output layer

    for l in range(1, len(activ_funcs)+1):
        input_tensor = layers_dict[l-1] # output of previous layer after activation is applied
            
        weights_key = 'w' + str(l)
        biases_key = 'b' + str(l)
        activation_function = activ_funcs[l-1]
        layer_type = layer_types[l]

        if(l == len(activ_funcs)):
            activation_function = 'linear'
        
        generic_layer = get_generic_layer(input_tensor, weights[weights_key], biases[biases_key], layer_type)
        layers_dict[str(l)+'i'] = generic_layer # wa + b before activation function is applied; e.g. layers_dict['2'] = f(layers_dict['2i'])
        
        layer = get_activation_function_layer(generic_layer, activation_function)
        layers_dict[str(l)+'f'] = layer
        
        mask_layer = get_mask_layer(layer)
        layers_dict[str(l)+'m'] = mask_layer
        
        resulting_layer = tf.math.multiply(layer, mask_layer)
        layers_dict[l] = resulting_layer
        
    return layers_dict

# creates a generic layer without activation function
def get_generic_layer(x, w, b, layer_type='ff'):
    if(layer_type == 'ff'):
        tf_layer = tf.add(tf.matmul(x, w), b) # linear layer
    else:
        raise Exception('Invalid layer type.')

    return tf_layer

# creates an activation function layer
def get_activation_function_layer(tf_layer, activ_fun):
    if(activ_fun == 'sigmoid'):
        tf_layer = tf.nn.sigmoid(tf_layer)
    elif(activ_fun == 'softmax'):
        tf_layer = tf.nn.softmax(tf_layer)
    elif(activ_fun == 'relu'): # not fully tested yet
        tf_layer = tf.nn.relu(tf_layer)

    return tf_layer

# creates a mask layer
def get_mask_layer(tf_layer):
    if(len(tf_layer.shape) == 1):
        mask_length = tf_layer.shape[0]
    else:
        mask_length = tf_layer.shape[1]
        
    mask_layer = tf.Variable(tf.ones(mask_length), trainable=False)
    return mask_layer

# layers size and arr_init define the initialization configuration
# Example output: For prefix 'o', o1 corresponds for Input layer, o2 for hidden layer 1, etc
def get_arrdict(layer_sizes, arr_init, prefix):
    dict = {}
    
    n_layers = len(layer_sizes)
    for l in range(0, n_layers):
        arr_name = prefix + str(l+1)
        
        if(arr_init == 'empty'):
            dict[arr_name] = np.array([], dtype=int)
        elif arr_init == 'range':
            n = layer_sizes[l]
            dict[arr_name] = np.array(range(n))

    return dict

# layers size and var_init define the architecture and initialization configuration of weights and biases in the network
# variable type is either weights (2D) or biases (1D) and prefix determines name of resulting variables
# Example output: For prefix 'w', w1 corresponds for weights between input and first hidden layers, etc.
def get_vardict(layer_sizes, var_init, var_type, prefix, activ_funcs, init_reduction='fan_in', seed=1234):
    dict = {}
    
    n_layers = len(layer_sizes)
    for l in range(1, n_layers):
        var_name = prefix + str(l)
        n_origin = layer_sizes[l-1]
        n_dest = layer_sizes[l]
        
        if(var_type == 'bias'):
            var_shape = [n_dest]
        elif(var_type == 'weight'):
            var_shape = [n_origin, n_dest]

        if(var_init == 'norm'):
            std_dev = 1.0
            if init_reduction == 'fan_in': #  scale std_dev to encourage values to be ~ favorable intervals, e.g. [-2, 2]
                # scale down by square root of destination layer size
                std_dev = (1.0/np.sqrt(n_dest))
                if activ_funcs[l-1] in ('sigmoid', 'relu'): 
                     std_dev *= 2

            dict[var_name] = tf.Variable(tf.random_normal(var_shape, stddev=std_dev, seed=seed))
        elif(var_init == 'zeros'):
            dict[var_name] = tf.Variable(tf.zeros(var_shape, dtype=tf.float32))

    return dict

# returns a reduced architecture (layer_sizes) equal to one after tuning is over
def reduce_architecture(layer_sizes, tuning_step, epochs, k_selected, n_tuned_layers, start_layer=1):
    print('Original architecture: {0}'.format(layer_sizes))
    total_k_selected = (n_tuned_layers * k_selected) * np.floor(epochs / tuning_step) # total number of neurons to be removed

    for l in range(n_tuned_layers):
        layer_sizes[start_layer+l] -= int((total_k_selected / n_tuned_layers))
    
    print('Reduced architecture: {0}'.format(layer_sizes))
    
    return layer_sizes

# reads data
def read_dataset(dataset_name='mnist', one_hot_encoding=True, noise_ratio=0, scaling_type='minmax', seed=1234):
    # parameters used on dataset-specific basis
    feature_scaling = False
    noise_type = 'zeros' # binomial noise is default

    if dataset_name == 'mnist':
        mnist = read_data_sets('../data/MNIST_data/', one_hot=one_hot_encoding)

        X_tr, Y_tr = mnist.train.images, mnist.train.labels
        X_val, Y_val = mnist.validation.images, mnist.validation.labels
        X_ts, Y_ts = mnist.test.images, mnist.test.labels
    elif dataset_name == 'psychencode':
        psychencode_filename = '../data/psychencode/DSPN_bpd_large/datasets/bpd_data1.mat'
        psychencode_data = loadmat(psychencode_filename)
        
        X_tr, Y_tr = psychencode_data['X_Gene_tr'], psychencode_data['X_Trait_tr']
        X_val, Y_val = None, None
        X_ts, Y_ts = psychencode_data['X_Gene_te'], psychencode_data['X_Trait_te']
    
        select_correlated_cols = True
        N = 932

        if(select_correlated_cols):
            X_tr, X_ts = get_correlated_features(X_tr, Y_tr, X_ts, N)

        if one_hot_encoding == False:
            Y_tr = np.argmax(Y_tr, axis=1)
            Y_ts = np.argmax(Y_ts, axis=1)
    elif 'diabetes' in dataset_name:
        if dataset_name == 'diabetes':
            diabetes_filename = '../data/csv_data/diabetes_data_processed.csv'
        if dataset_name == 'diabetes_SMOTE':
            diabetes_filename = '../data/csv_data/diabetes_data_smote_manoj_etal_processed.csv'

        print(diabetes_filename)
        diabetes_data =  np.genfromtxt(diabetes_filename, delimiter=',', skip_header=1) # diabetes shape: (101767, 36)

        X = diabetes_data[:, 0:-1] # select all data columns
        Y = diabetes_data[:, -1].astype(int) # select labels (last) column

        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.25, random_state=seed)

        Y_ts = np.maximum(Y_ts, 0, Y_ts) # max w/ 0 to fix artifact in data where some y vales < 0
        
        feature_scaling = True
        noise_type = 'norm' # Gaussian noise
    elif dataset_name in ('xor', 'moons'):
        N = 4000 # number of data points
        D = 8 # number of noisy features
        
        if dataset_name == 'moons':
            from sklearn.datasets import make_moons
            X_signal, Y = make_moons(n_samples=N, noise=0.1)
            X = np.random.normal(0.0, scale=1.0, size=(N, D+2)) # D cols of gaussian noise; first 2 cols to include signal
            noise_type = 'norm'
        elif dataset_name == 'xor':
            bern_sample1 = np.random.choice(2, N, p=[0.5, 0.5]) # "fair" bernouilli samples
            bern_sample2 = np.random.choice(2, N, p=[0.5, 0.5])
            Y = np.logical_xor(bern_sample1, bern_sample2).astype(int)

            X_signal = np.concatenate((bern_sample1, bern_sample2)).reshape(2, N).T
            X = np.random.choice(2, (N, D+2), p=[0.5, 0.5]) # D cols of bernoulli "fair" noise; first 2 cols to include signal
        
        X[:, 0:2] = X_signal
        
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.25, random_state=seed) # Final split: 60-20-20%
    elif 'AML' in dataset_name:
        task_id = int(re.search('AML_(\d)', dataset_name).group(1))
        
        drug_filename = '../data/AML-Drug-Sensitivity/S10_Drug_Responses.csv' # drug one-hot encodings included in input data
        drug_data = np.genfromtxt(drug_filename, usecols=(0, 1, 2), deletechars="~!@#$%^&*()=+~\|]}[{'; /?.>,<", delimiter=',', skip_header=1, dtype=[('inhibitor', 'U25'), ('lab_id', 'U10'), ('ic50', 'f8')], encoding='utf-8')

        drug_list = np.unique(drug_data['inhibitor'])
        drug_ind = {}
        for d in range(len(drug_list)):
            drug_ind[drug_list[d]] = d # index of each drug in the list of unique drugs; to be used when generating final dataset
        
        n_drugs = len(drug_list)
        
        X_cols = n_drugs # baseline

        if task_id < 4: # expression profiles included in tasks 1-3
            expr_filename = '../data/AML-Drug-Sensitivity/S8_Gene_Counts_RPKM.csv'
        
            expr_cols = 451
            expr_samples = np.genfromtxt(expr_filename, usecols=np.arange(2, expr_cols), max_rows=1, names=True, deletechars="~!@#$%^&*()=+~\|]}[{'; /?.>,<", delimiter=',', encoding='utf-8').dtype.names
            expr_genes = np.genfromtxt(expr_filename, usecols=(0, 1), names=True, dtype=[('Gene', 'U25'), ('Symbol', 'U25')], delimiter=',')
        
            expr_data = np.genfromtxt(expr_filename, usecols=np.arange(2, expr_cols), skip_header=1, delimiter=',')
            n_expr_genes = expr_data.shape[0];
            X_cols += n_expr_genes # include gene expression vectors
            
            expr_sample_ind = {} # index of a expr_samples is its col index in expression file; dict to be used when creating final dataset
            for s in range(len(expr_samples)):
                expr_sample_ind[expr_samples[s]] = s
                
        if task_id > 1: # variant matrix included in tasks 2-4
            var_filename = '../data/AML-Drug-Sensitivity/Variants_Samples_Impact_Matrix.csv' # S x G matrix IM; IM[s, g] = total inverted SIFT score in gene g in sample s; the higher value is, the more deleterious the impact in g in s
            
            var_cols = 3334
            var_genes = np.genfromtxt(var_filename, usecols=np.arange(2, var_cols), max_rows=1, names=True, deletechars="~!@#$%^&*()=+~\|]}[{'; /?.>,<", delimiter=',', encoding='utf-8').dtype.names
            var_samples = np.asarray(np.genfromtxt(var_filename, usecols=(0), names=True, dtype=[('labId', 'U10')], delimiter=',').tolist()).flatten()
            
            var_data = np.genfromtxt(var_filename, usecols=np.arange(2, var_cols), skip_header=1, delimiter=',')
            
            if task_id == 2: 
                var_data = np.sum(var_data, axis=1)
                X_cols += 1 # include singular vaiant deleteriousness count
            else: 
                X_cols += var_data.shape[1] # number of genes in which there are variants        
                
            var_sample_ind = {} # index of a var_samples is its col index in variants file; dict to be used when creating final dataset
            for s2 in range(len(var_samples)):
                var_sample_ind[var_samples[s2]] = s2
        
        drug_inds = []; expr_sample_inds = []; var_sample_inds = []; selected_points = []
        for i in range(drug_data.shape[0]):
            expr_include_point = False; var_include_point = False
            if task_id < 4 and drug_data['lab_id'][i] in expr_sample_ind.keys():
                expr_include_point = True
                
            if  task_id > 1 and drug_data['lab_id'][i] in var_sample_ind.keys():
                var_include_point = True
                
            include_point = expr_include_point
            if task_id > 1 and task_id <4:
                include_point = include_point and var_include_point
            elif task_id > 3:
                include_point = var_include_point
            
            if include_point:
                if task_id < 4:
                    expr_sample_inds.append(expr_sample_ind[drug_data['lab_id'][i]]) # used to select expression profile as part of input
                
                if task_id > 1:
                    var_sample_inds.append(var_sample_ind[drug_data['lab_id'][i]]) # used to select expression profile as part of input
                
                drug_inds.append(drug_ind[drug_data['inhibitor'][i]]) # used for to build one-hot encoded vector as part of input
                selected_points.append(i)
        
        X = np.zeros((len(drug_inds), X_cols))
        
        drug_col_start = 0 # drugs vector first, all tasks
        X[np.arange(X.shape[0]), drug_col_start + np.array(drug_inds)] = 1 # one-hot encoded drug vector block in input dataset
        
        if task_id < 4: # expression vectors second, if needed in task at hand
            expr_col_start = n_drugs; expr_col_end = expr_col_start + n_expr_genes
            X[:, expr_col_start:expr_col_end] = expr_data[:, expr_sample_inds].T # expression profiles block, tasks 1-3
            
        if task_id > 1 and task_id < 4:
            var_col_start = expr_col_start + n_expr_genes
            if task_id == 2: # drug vector > expression vector > variation (deleteriousness) singular, sum value
                var_col_end = var_col_start + 1
                X[:, var_col_start] = var_data[var_sample_inds] 
            elif task_id == 3: # drug vector > expression vector > variation (deleteriousness) vector
                var_col_end = var_col_start + var_data.shape[1]
                X[:, var_col_start:var_col_end] = var_data[var_sample_inds, :]
        elif task_id == 4: # drug vector > variation (deleteriousness) vector
            var_col_start = n_drugs; var_col_end = var_col_start + var_data.shape[1]
            X[:, var_col_start:var_col_end] = var_data[var_sample_inds, :] 
                
        Y = drug_data['ic50'][selected_points]
        
        threshold = 5
        Y[Y <= threshold] = 0; Y[Y > threshold] = 1; Y = Y.astype('int32') # binarize task
        
        print(X.shape)
        print(Y.shape)
        
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.25, random_state=seed) # Final split: 60-20-20%

        feature_scaling = True
        
    if feature_scaling:
        if scaling_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_type == 'standard':
            scaler = StandardScaler()
            
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_ts = scaler.transform(X_ts)
        print('{0} scaling done.'.format(scaling_type))
    
    if one_hot_encoding and dataset_name != 'mnist': # mnist labels are already one-hot encoded
            n_values = len(np.unique(Y_tr))
            Y_tr = np.eye(n_values)[Y_tr]
            Y_val = np.eye(n_values)[Y_val]
            Y_ts = np.eye(n_values)[Y_ts] 

    if noise_ratio > 0.0:
        if noise_type == 'zeros': # zeros noise
            X_tr[np.random.sample(size=X_tr.shape) < noise_ratio] = 0
        elif noise_type == 'norm': # Gaussian noise
            X_tr += np.random.normal(0, noise_ratio/2, X_tr.shape) # sd=noise_ratio/2 to cover ranges of values after minmax scaling appropriately
        
        print("Noise at ration {0} added.".format(noise_ratio))
        
    train = DataSet(X_tr, Y_tr)
    validation = DataSet(X_val, Y_val)
    test = DataSet(X_ts, Y_ts)

    return Datasets(train=train, validation=validation, test=test)  

# return N columns with highest correlation with the label in training data
def get_correlated_features(X_tr, Y_tr, X_ts, N):
    Y_tr = np.argmax(Y_tr, axis=1)
    corr = abs(np.apply_along_axis(pearsonr, 0, X_tr, Y_tr)[0])
    
    # select N columns with highest correlation with Y_tr
    selected_columns = np.flip(np.argsort(corr))[0:N]
    X_tr_selected = X_tr[:, selected_columns]
    X_ts_selected = X_ts[:, selected_columns]

    return X_tr_selected, X_ts_selected

# batch_index starts at 0
def get_next_batch(X_tr, Y_tr, batch_index, batch_size, seed=1234, shuffle=True):
    start = batch_index * batch_size
    end = start + batch_size - 1
    
    if end > X_tr.shape[0]:
        end = X_tr.shape[0] - 1
            
    batch_x = X_tr[start:end, :]
    batch_y = Y_tr[start:end, :]
    
    return batch_x, batch_y

# sets seed of helper function
def set_seed(seed=1234):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed=seed)
    print('Seeds in helper functions set to {}'.format(seed))

# unpacks values from returned hyperopt spaces. 
# In some instances, returned values are each in a list (even for singular values)
# used for space_eval function for better representation
def unpack_dict(input_dict):
    unpacked_dict = {}
    
    for key, value in list(input_dict.items()):
        if(type(value) is list):
            unpacked_dict[key] = value[0]
        else:
            unpacked_dict[key] = value
        
    return unpacked_dict

# to get best result with its corresponding hyperparameter from a dictionary returned by hyperopt
# t is a Trials object after the execution of hyperopt's fmin()
def get_best_result(t, hp_space, metric='accuracy'):
    best_metric_value = 0
    best_trial_result = None
    best_trial_hyperparam_space = {}
    for trial in t.trials:
        try:
            if (trial['result'][metric] > best_metric_value):
                best_metric_value = trial['result'][metric]
                best_trial_result = trial['result']
                best_trial_hyperparam_space = unpack_dict(trial['misc']['vals'])      
        except:
            print('Error with a hyperparameter space occurred.')
            continue

    best_trial_result.update(space_eval(hp_space, best_trial_hyperparam_space)) # merge dictionaries
    return best_trial_result # returns merged dictionaries (results+hyperparameter space)

# to write weight, bias dict of matrices into a text file 
def save_vardict_to_file(filebasename_prefix, vardict, epoch, seed=1234, dict_name='weight', pickling=False, sep='\t'):
    now = datetime.datetime.now()
    
    if dict_name == 'bias':
        suffix = 'es'
    elif dict_name == 'weight':
        suffix = 's'
    else:
        raise Exception('Invalid dict_name.')
    
    filebasename = filebasename_prefix + '_ep' + str(epoch) + '_sd' + str(seed) + '_' + dict_name + suffix + '_' + str(now.isoformat())
    all_weights_str = ''

    # check for pickling
    if pickling:
        
        pickle_file = open(filebasename + '.pkl', 'wb')
        pickle.dump(vardict, pickle_file)

    keys_ordered = sorted(vardict.keys(), reverse=True)
    for key in keys_ordered:
        layer_weights_str = ''
        weights = vardict[key]

        if(dict_name == 'bias'):
            layer_weights_str = ''
            for ind in range(weights.shape[0]):
                layer_weights_str = layer_weights_str + str(weights[ind])
                
                if ind != (weights.shape[0]-1):
                    layer_weights_str = layer_weights_str + sep
                else:
                    layer_weights_str = layer_weights_str + '\n'
        elif(dict_name == 'weight'):
            for row in range(weights.shape[0]):
                row_str = ''
                for col in range(weights.shape[1]):
                    row_str = row_str + str(weights[row, col])
                
                    if col != (weights.shape[1]-1):
                        row_str = row_str + sep
    
                layer_weights_str = layer_weights_str + row_str + '\n'
        
        all_weights_str = all_weights_str + layer_weights_str
    
    output_file = open(filebasename + '.txt', 'w+')
    output_file.write(all_weights_str)
    output_file.close()
    
# removes outliers and returns values in [1st-99th] percentile along axis
def percentile_input_matrix(input_matrix, bottom_percentile=1, top_percentile=99, axis=1):
    n_items = input_matrix.shape[axis]
    single_percentile = n_items/100
    
    bottom_percentile = int(np.ceil((bottom_percentile/100)*n_items)) # index of bottom percentile
    top_percentile = int(np.ceil((top_percentile/100)*n_items))
    
    input_matrix = np.sort(input_matrix, axis=axis)
    
    if(axis == 0):
        input_matrix = input_matrix[bottom_percentile:top_percentile, :]
    elif(axis == 1):
        input_matrix = input_matrix[:, bottom_percentile:top_percentile]
    
    return input_matrix
