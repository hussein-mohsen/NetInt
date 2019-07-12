import random

import numpy as np
#from numpy import dtype, shape

import tensorflow as tf
import networkx as nx
import collections

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from numpy.random.mtrand import shuffle

from scipy.io import loadmat
from scipy.stats import pearsonr

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

from scipy import stats
from scipy.stats import powerlaw

from helper_objects import DataSet
from tensorflow.python.framework import dtypes

from hyperopt import space_eval

seed = 1234
epsilon = 0.00001

# Create the MLP
# x is the input tensor, activ_funs is a list of activation functions
# weights and biases are dictionaries with keys = 'w1'/'b1', 'w2'/'b2', etc.
def multilayer_perceptron(x, weights, biases, activ_funcs, layer_types):
    layers = []
    
    for l in range(len(activ_funcs)):
        if(l == 0):
            input_tensor = x
        else:
            input_tensor = layers[l-1]
            
        weights_key = 'w' + str(l+1)
        biases_key = 'b' + str(l+1)
        activation_function = activ_funcs[l]
        layer_type = layer_types[l]

        if(l == len(activ_funcs)-1):
            activation_function = 'linear'
            
        layer = get_layer(input_tensor, weights[weights_key], biases[biases_key], activation_function, layer_type)
        layers.append(layer)

    print(layers)
    return layers[-1] # return the last tensor, i.e. output layer

# creates TF layers
def get_layer(x, w, b, activ_fun, layer_type='ff'):
    if(layer_type == 'ff'):
        tf_layer = tf.add(tf.matmul(x, w), b) # linear layer
        
        if(activ_fun == 'sigmoid'):
            tf_layer = tf.nn.sigmoid(tf_layer)
        elif(activ_fun == 'softmax'):
            tf_layer = tf.nn.softmax(tf_layer)
        elif(activ_fun == 'relu'): # not fully tested yet
            tf_layer = tf.nn.relu(tf_layer)
    else:
        raise Exception('Invalid layer type.')
    
    return tf_layer

# calculates KL divergence of two distributions
def kl_div(empirical, target):
    if(abs(empirical.sum()-1) > 0.05 or abs(target.sum()-1) > 0.05):
        print("Warning: one or more distributions do not sum up to 1")
        
    kl_div_value = (empirical * np.log(empirical/target)).sum()
    return kl_div_value

# calculate KS test to compare distance between CDFs
def ks_test(empirical, target_distribution='norm', metric='D'):
    if(abs(empirical.sum()-1) > 0.05):
        print("Warning: empirical distribution does not sum up to 1")

    # set cdf arguments
    if target_distribution == 'norm':
        distribution_args = (0, 1) # (loc, scale)
    elif target_distribution == 'powerlaw':
        distribution_args = (1.5, 0, 1) # (a, loc, scale)

    ks_results = stats.kstest(empirical, target_distribution, args=distribution_args)
    
    if metric == 'D':
        return ks_results[0]
    elif metric == 'p_value':
        return ks_results[1]

# scales values s.t. sum = 1
def totality_scale(values):
    total = values.sum()
    return ((values/total) + epsilon)

# scale all values to [0,1]
def minmax_scale(values):
    values = (values - values.min())/(values.max()-values.min())
    return (values + epsilon)

# scales empirical matrix to [0,1] per scale_type
def scale_input_matrix(input_matrix, shift_type='min', scale_type='totality', axis=1):
    if scale_type == 'totality':
        if shift_type == 'min':
            input_matrix += abs(input_matrix.min())
        elif shift_type == 'abs':
            input_matrix = abs(input_matrix)
        else:
            raise Exception('Invalid shift type.')
    
        input_matrix = np.apply_along_axis(totality_scale, axis, input_matrix)
    elif scale_type == 'minmax':
        input_matrix = np.apply_along_axis(minmax_scale, axis, input_matrix)

    return input_matrix

# scales input vector to [0,1] per scale_type
def scale_input_vector(input_vector, scale_type='totality', shift_type='min'):
    if scale_type == 'totality':
        if shift_type == 'min':
            input_vector += abs(input_vector.min())
        elif shift_type == 'abs':
            input_vector = abs(input_vector)
        else:
            raise Exception('Invalid shift type.')
        
        input_vector = totality_scale(input_vector)
    elif scale_type == 'minmax':
        input_vector = minmax_scale(input_vector)

    return input_vector

# returns a pmf from of a vectors histogram
def calculate_histogram_pmf(vector, n_bins=10):
    vector_histogram, bins = np.histogram(vector, bins=n_bins)
    return totality_scale(vector_histogram)

# Calculate KL div between scaled weights (per row) with scaled target distribution
def calculate_scaled_kl_div(input_matrix, seed=seed,
                            shift_type= 'min', scale_type='totality',
                            target_distribution='norm', axis=1):
    
    input_matrix = scale_input_matrix(input_matrix, shift_type=shift_type, scale_type=scale_type, axis=axis) # scaling
    input_matrix = np.apply_along_axis(calculate_histogram_pmf, axis, input_matrix) # histograms

    if target_distribution == 'norm':
        target_dist = np.random.normal(1, 0.1, (1, input_matrix.shape[axis]))
    elif target_distribution == 'powerlaw': # power law distribution
        a = 1.5
        x = np.linspace(powerlaw.ppf(0.01, a), powerlaw.ppf(0.99, a), input_matrix.shape[axis])
        target_dist = np.asarray(powerlaw.pdf(x=x, a=a, loc=0, scale=1))
    else:
        raise Exception('Invalid target distribution.')
    
    target_dist = scale_input_vector(target_dist, shift_type=shift_type, scale_type=scale_type) # scaling
    target_dist = calculate_histogram_pmf(target_dist) # histogram

    print("Target distribution inner sum: "+str(target_dist.sum()))
    
    kl_values = np.apply_along_axis(kl_div, axis, input_matrix, target_dist)
    return kl_values
    
# Calculate KS distance (D or p-)value between scaled weights (per row) with scaled target distribution
def calculate_ks_distance(input_matrix, seed=seed, shift_type='min', scale_type='totality',
                          target_distribution='norm', ks_metric='D', axis=1):
    
    input_matrix = scale_input_matrix(input_matrix, shift_type=shift_type, scale_type=scale_type, axis=axis)
    
    if target_distribution not in ['norm', 'powerlaw']:
        raise Exception('Invalid target distribution.')
     
    ks_values = np.apply_along_axis(ks_test, axis, input_matrix, target_distribution, ks_metric)
    return ks_values

# calculates distribution distance (of all rows) per tuning type
def calculate_distance_values(weights, tuning_type='kl_div', shift_type='min', scale_type='totality',
                              target_distribution='norm', ks_metric='D', axis=1):

    if tuning_type == 'kl_div':
        distance_values = calculate_scaled_kl_div(weights, shift_type=shift_type, scale_type=scale_type,
                                                  target_distribution=target_distribution, axis=axis)
    elif tuning_type == 'ks_test':
        distance_values = calculate_ks_distance(weights, shift_type=shift_type, scale_type=scale_type,
                                                target_distribution=target_distribution, ks_metric=ks_metric, 
                                                axis=axis)
        
    return distance_values

# calculates KL divergence from a target distribution for incoming and outcoming weight distributions
# KL calculated after min-max scaling to [0, 1] + eps; returns averaged incoming and outcoming scores for each neuron
def calculate_layer_distance_values(weights_dict, layer_index, 
                                    shift_type='min', scale_type='totality', 
                                    target_distribution='norm', tuning_type='kl_div', 
                                    ks_metric='D'):    
    if layer_index > len(weights_dict): # output layer or erroneous index
        raise Exception('Layer index is out of bounds.')
    else:
        outcoming_weights = weights_dict['w'+str(layer_index)]        
        distance_values = calculate_distance_values(outcoming_weights, tuning_type=tuning_type, 
                                                    shift_type=shift_type, scale_type=scale_type, 
                                                    target_distribution=target_distribution, axis=1)
        
        #return distance_values
    
        if layer_index > 1:
            incoming_weights = weights_dict['w'+str(layer_index-1)]
            incoming_distance_values = calculate_distance_values(incoming_weights, tuning_type=tuning_type, shift_type=shift_type,
                                                                 scale_type=scale_type, target_distribution=target_distribution, 
                                                                 axis=0)

            distance_values = (distance_values + incoming_distance_values) / 2
            
    return distance_values

# helper function to get indices of layer at index in a graph
def get_layer_inds(boundaries, index):
    if(index < 1 or index > len(boundaries)):
        raise Exception('Index is out of bounds.')
    elif(index == 1): # input layer
        start = 0
    else: # hidden layers
        start = boundaries[index-2]
    
    end = boundaries[index-1] -1
    
    return start, end

# gets indices to be tuned (i.e. turned off and replaced by 0 values)
# available indices are the ones not tuned prior to selection   
# tuning_type: 'centrality' (default: betweenness centrality) 
#              'kl_div' (default: with Gaussian)   
#              'random': 
def get_off_inds(weights_dict, avail_inds, layer_index, input_list=[], 
                 k_selected=4, tuning_type='centrality', dt=[('weight', float)], 
                 shift_type='min', scale_type='totality', target_distribution='norm'):    
    if tuning_type == 'random': # random selection of indices
        select_inds = random.sample(range(len(avail_inds)), k_selected) # indices within avail_inds to be turned off        
    else:
        if tuning_type == 'centrality': # sorted centrality-based selection
            weight_graph, layer_boundaries = create_weight_graph(weights_dict, layer_index)
            weight_graph = weight_graph.astype(dt)
            weight_G = nx.from_numpy_matrix(weight_graph) # create graph object
            
            # calculate centrality measure values
            print("Calculating centrality measures...")
            values = np.array(list(nx.betweenness_centrality(weight_G, k=7, weight='weight').values()))
        elif tuning_type == 'kl_div' or tuning_type == 'ks_test':
            # calculate KL divergence from a target distribution (default: Gaussian)
            print("Calculating distribution distance values per {0}...".format(tuning_type))
            values = calculate_layer_distance_values(weights_dict, layer_index, tuning_type=tuning_type,
                                                     shift_type=shift_type, scale_type=scale_type, 
                                                     target_distribution=target_distribution)
        else:
            raise Exception('Invalid tuning type value.')
            
        # select nodes with lowest k_selected to tune
        values = values[avail_inds]
        inds = np.argsort(values)
        select_inds = inds[0:k_selected]
    
    off_inds = avail_inds[select_inds]
    return off_inds # return array of indices to be tuned

# helper function that pads a matrix to create an NxN graph used to
# create weight graphs on which centrality measures are calculated
def pad_matrix(input_matrix):
    rows_increment = max(0, input_matrix.shape[1]-input_matrix.shape[0])
    cols_increment = max(0, input_matrix.shape[0]-input_matrix.shape[1])

    input_matrix = np.pad(input_matrix, pad_width=((0, rows_increment), (0, cols_increment)), mode='constant', constant_values=(0, 0))
    return input_matrix

# creates a weight graph at an input layer
# Layer indexing starts at 1; numpy indexing starts at 0
def create_weight_graph(weights_dict, layer):
    if layer == 1: # input layer: edge case where weight graph is made of one layer
        return pad_matrix(weights_dict['w'+str(layer)])
    elif layer > len(weights_dict): # output layer or erroneous index
        raise Exception('Layer index is out of bounds.')
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

# helper function that turns off neurons at off_indices (0 value assignment)
# returns a tf tensor
def tune_weights(off_indices, current_weights, layer):
    current_weights['w'+str(layer)][off_indices, :] = 0 # turn off connections outcoming from off_indices neurons in the layer
    print("Outcoming connections at layer {} tuned.".format(layer))
     
    if(layer > 1):
        current_weights['w'+str(layer-1)][:, off_indices] = 0 # turn off connections incoming to off_indices neurons in the layer
        print("Incoming connections at layer {} tuned.".format(layer))
        
    return tf.convert_to_tensor(current_weights['w'+str(layer)], dtype=tf.float32)

# reads data
def read_dataset(dataset_name='mnist', minmax_scaling=False):
    if(dataset_name == 'mnist'):
        mnist = read_data_sets('../data/MNIST_data/', one_hot=True)

        X_tr, Y_tr = mnist.train.images, mnist.train.labels
        X_val, Y_val = mnist.validation.images, mnist.validation.labels
        X_ts, Y_ts = mnist.test.images, mnist.test.labels
    elif(dataset_name == 'psychencode'):
        psychencode_filename = '../data/psychencode/DSPN_bpd_large/datasets/bpd_data1.mat'
        psychencode_data = loadmat(psychencode_filename)
        
        X_tr, Y_tr = psychencode_data['X_Gene_tr'], psychencode_data['X_Trait_tr']
        X_val, Y_val = None, None
        X_ts, Y_ts = psychencode_data['X_Gene_te'], psychencode_data['X_Trait_te']
    
        select_correlated_cols = True
        N = 932
        
        if(select_correlated_cols):
            X_tr, X_ts = get_correlated_features(X_tr, Y_tr, X_ts, N)
    elif(dataset_name == 'diabetes'):
        diabetes_filename = '../data/csv_data/diabetes_data_processed.csv'
        diabetes_data =  np.genfromtxt(diabetes_filename, delimiter=',') # diabetes shape: (101767, 36)

        X = diabetes_data[:, 0:-1]
        Y = diabetes_data[:, -1].astype(int) 
        
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.25, random_state=seed)
        
        n_values = len(np.unique(Y_tr))
        Y_ts = np.maximum(Y_ts, 0, Y_ts) # max w/ 0 to fix artifact in data where some y vales < 0

        Y_tr = np.eye(n_values)[Y_tr]
        Y_val = np.eye(n_values)[Y_val]
        Y_ts = np.eye(n_values)[Y_ts] 
    
    if minmax_scaling:
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.fit_transform(X_val)
        X_ts = scaler.transform(X_ts)
        print("Minmax scaling done.")
    
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
    
# sets seed of helper function
def set_seed(seed=1234):
    tf.set_random_seed(seed)
    np.random.seed(seed=seed)
    random.seed(seed)
    print("Seeds in helper functions set to {}".format(seed))
    
# epochs start at 1, index in data at 0
# Method and code structure from mnist.next_batch
def get_next_even_batch(X_tr, Y_tr, start, batch_size, epoch, seed=seed, shuffle=True):
    end = start + batch_size
            
    if end > X_tr.shape[0]:
        N_remaining_points = X_tr.shape[0] - start
        remaining_points = X_tr[start:X_tr.shape[0]]
        remaining_labels = Y_tr[start:Y_tr.shape[0]]
        
        if shuffle == True:
            data_order = np.arange(X_tr.shape[0])
            np.random.shuffle(data_order)

            X_tr = X_tr[data_order]
            Y_tr = Y_tr[data_order]
        
        N_new_points = batch_size - N_remaining_points
        new_points = X_tr[0:N_new_points]
        new_labels = Y_tr[0:N_new_points]
        
        batch_x = np.concatenate((remaining_points, new_points), axis=0)
        batch_y = np.concatenate((remaining_labels, new_labels), axis=0)
        next_start = N_new_points
    else:
        batch_x = X_tr[start:end]
        batch_y = Y_tr[start:end]
        next_start = end % X_tr.shape[0]

    return batch_x, batch_y, next_start

# batch_index starts at 0
def get_next_batch(X_tr, Y_tr, batch_index, batch_size, seed=seed, shuffle=True):
    start = batch_index * batch_size
    end = start + batch_size - 1
    
    if end > X_tr.shape[0]:
        end = X_tr.shape[0] - 1
            
    batch_x = X_tr[start:end, :]
    batch_y = Y_tr[start:end, :]
    
    return batch_x, batch_y

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
def get_vardict(layer_sizes, var_init, var_type, prefix):
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
            dict[var_name] = tf.Variable(tf.random_normal(var_shape))
        elif(var_init == 'zeros'):
            dict[var_name] = tf.Variable(tf.zeros(var_shape, dtype=tf.float32))

    return dict

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
