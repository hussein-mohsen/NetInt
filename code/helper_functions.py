import random
import datetime
import pickle

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

from helper_objects import DataSet
from tensorflow.python.framework import dtypes

from hyperopt import space_eval
from setuptools.dist import Feature

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
def kl_div(empirical, target_dist):
    if(abs(empirical.sum()-1) > 0.05 or abs(target_dist.sum()-1) > 0.05):
        print("Warning: one or more distributions do not sum up to 1.")

    kl_div_value = (empirical * np.log((empirical + epsilon)/(target_dist + epsilon))).sum()
    return kl_div_value

# calculate KS test to compare distance between CDFs of empirical and target_dist samples
def ks_test(empirical, target_dist, metric='D'):        
    ks_results = stats.ks_2samp(empirical, target_dist)
    
    if metric == 'D':
        return ks_results[0]
    elif metric == 'p_value':
        return ks_results[1]

# scales values s.t. sum = 1
def totality_scale(values):
    total = values.sum() + epsilon
    return (values/total)

# scale all values to [0,1]
def minmax_scale(values):
    values = (values - values.min())/((values.max() - values.min() + epsilon))
    return values

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

# scales empirical matrix to [0,1] per scale_type
def scale_input_matrix(input_matrix, shift_type='min', scale_type='minmax', axis=1):
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
def scale_input_vector(input_vector, scale_type='minmax', shift_type='min'):
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
def calculate_scaled_kl_div(input_matrix, seed=1234,
                            shift_type= 'min', scale_type='minmax',
                            target_distribution='norm', axis=1):

    input_matrix = scale_input_matrix(input_matrix, shift_type=shift_type, scale_type=scale_type, axis=axis) # scaling
    input_matrix = np.apply_along_axis(calculate_histogram_pmf, axis, input_matrix) # histograms

    if target_distribution == 'norm':
        target_dist = np.random.normal(1, 0.1, (1, input_matrix.shape[axis]))
    elif 'powerlaw' in target_distribution: # power law and inverse power law distribution
        target_dist = np.random.power(a=0.35, size=(input_matrix.shape[axis], ))
    else:
        raise Exception('Invalid target distribution.')
    
    target_dist = scale_input_vector(target_dist, scale_type=scale_type) # scaling
    if 'inv' in target_distribution: # inverted power law distribution; higher density on higher weight values
        target_dist = (np.max(target_dist) - target_dist) + epsilon # invert distribution after minmax, i.e. 1-each value in scaled sample vector

    # calculate histograms of empirical data and target_dist
    input_matrix = np.apply_along_axis(calculate_histogram_pmf, axis, input_matrix) # histograms
    target_dist = calculate_histogram_pmf(target_dist) # histogram

    kl_values = np.apply_along_axis(kl_div, axis, input_matrix, target_dist)
    return kl_values
    
# Calculate KS distance (D or p-)value between scaled weights (per row) with scaled target distribution
def calculate_ks_distance(input_matrix, seed=1234, shift_type='min', scale_type='minmax',
                          target_distribution='norm', ks_metric='D', axis=1):
    
    if target_distribution == 'norm':
        target_dist = np.random.normal(0, 0.1, size=(input_matrix.shape[axis], ))
    elif 'powerlaw' in target_distribution: # power law and inverse power law distribution
        target_dist = np.random.power(a=0.35, size=(input_matrix.shape[axis], ))
    else:
        raise Exception('Invalid target distribution.')
    
    target_dist = scale_input_vector(target_dist, scale_type=scale_type) # scaling
    if 'inv' in target_distribution: # inverted power law distribution; higher density on higher values
        target_dist = (np.max(target_dist) - target_dist) + epsilon # invert distribution after minmax, i.e. 1-each value in scaled sample vector
        
    ks_values = np.apply_along_axis(ks_test, axis, input_matrix, target_dist, ks_metric)
    return ks_values

# calculates distribution distance (of all rows) per tuning type
def calculate_distance_values(weights, tuning_type='kl_div', shift_type='min', scale_type='minmax',
                              target_distribution='norm', ks_metric='D',percentiles=False, axis=1):
    if(percentiles): # exclude outliers and keep values in [1st, 99th] percentiles
        weights = percentile_input_matrix(weights, 1, 99, axis=axis)

    weights = scale_input_matrix(weights, scale_type=scale_type, axis=axis) # scaling

    if 'inv' in target_distribution and scale_type != 'minmax':
        scale_type = 'minmax' # inverse distributions are based on minmax scaling (1 - original distirbution)
        print('Note: Inverted distribution to be calculated. Scaling set to minmax.')
        
    if tuning_type == 'kl_div':
        distance_values = calculate_scaled_kl_div(weights, scale_type=scale_type, target_distribution=target_distribution, 
                                                  axis=axis)
    elif tuning_type == 'ks_test':
        distance_values = calculate_ks_distance(weights, scale_type=scale_type, target_distribution=target_distribution, 
                                                ks_metric=ks_metric, axis=axis)

    return distance_values

# calculates KL divergence from a target distribution for incoming and outcoming weight distributions
# KL calculated after min-max scaling to [0, 1] + eps; returns averaged incoming and outcoming scores for each neuron
def calculate_layer_distance_values(weights_dict, layer_index, 
                                    shift_type='min', scale_type='minmax', 
                                    target_distribution='norm', tuning_type='kl_div', 
                                    ks_metric='D', percentiles=False):    
    if layer_index > len(weights_dict): # output layer or erroneous index
        raise Exception('Layer index is out of bounds.')
    else:
        outcoming_weights = weights_dict['w'+str(layer_index)]        
        distance_values = calculate_distance_values(outcoming_weights, tuning_type=tuning_type,shift_type=shift_type, 
                                                    scale_type=scale_type, target_distribution=target_distribution, 
                                                    percentiles=percentiles, axis=1)

    
        if layer_index > 1:
            incoming_weights = weights_dict['w'+str(layer_index-1)]
            incoming_distance_values = calculate_distance_values(incoming_weights, tuning_type=tuning_type, shift_type=shift_type,
                                                                 scale_type=scale_type, target_distribution=target_distribution, 
                                                                 percentiles=percentiles, axis=0)

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
def get_off_inds(weights_dict, avail_inds, off_inds, layer_index, input_list=[], 
                 k_selected=4, tuning_type='centrality', dt=[('weight', float)], 
                 shift_type='min', scale_type='minmax', target_distribution='norm',
                 percentiles=False):
    if(len(avail_inds) == 0):
        select_inds = [-1]
        print("Warning: no more neurons to tune.")
    else:
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
                
                # select the nodes corresponding to the layer since the graph = its nodes + those of preceding 
                # and following layers
                layer_start = layer_boundaries[0]; layer_end = layer_boundaries[1]
                values = values[layer_start:layer_end]
            elif tuning_type == 'kl_div' or tuning_type == 'ks_test':
                # calculate KL divergence from a target distribution (default: Gaussian)
                print("Calculating distribution distance values per {0}...".format(tuning_type))
                values = calculate_layer_distance_values(weights_dict, layer_index, tuning_type=tuning_type,
                                                         shift_type=shift_type, scale_type=scale_type, 
                                                         target_distribution=target_distribution,
                                                         percentiles=percentiles)
            else:
                raise Exception('Invalid tuning type value.')
    
            # select nodes with lowest k_selected to tune
            inds = np.argsort(values)
            inds = inds[~np.in1d(inds, off_inds)] # remove current off_inds
            select_inds = inds[0:k_selected]

    return select_inds # return array of indices to be tuned

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
def read_dataset(dataset_name='mnist', one_hot_encoding=True, seed=1234):
    minmax_scaling = False
    if(dataset_name == 'mnist'):
        mnist = read_data_sets('../data/MNIST_data/', one_hot=one_hot_encoding)
        
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

        if one_hot_encoding == False:
            Y_tr = np.argmax(Y_tr, axis=1)
            Y_ts = np.argmax(Y_ts, axis=1)
            
    elif(dataset_name == 'diabetes'):
        diabetes_filename = '../data/csv_data/diabetes_data_processed.csv'
        diabetes_data =  np.genfromtxt(diabetes_filename, delimiter=',', skip_header=1) # diabetes shape: (101767, 36)

        X = diabetes_data[:, 0:-1] # select all data columns
        Y = diabetes_data[:, -1].astype(int) # select labels (last) column

        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.25, random_state=seed)
        
        n_values = len(np.unique(Y_tr))
        Y_ts = np.maximum(Y_ts, 0, Y_ts) # max w/ 0 to fix artifact in data where some y vales < 0
        
        minmax_scaling = True
        
        if one_hot_encoding:
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
    random.seed(seed)
    np.random.seed(seed=seed)
    print("Seeds in helper functions set to {}".format(seed))
    
# epochs start at 1, index in data at 0
# Method and code structure from mnist.next_batch
def get_next_even_batch(X_tr, Y_tr, start, batch_size, epoch, seed=1234, shuffle=True):
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
def get_next_batch(X_tr, Y_tr, batch_index, batch_size, seed=1234, shuffle=True):
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
def get_vardict(layer_sizes, var_init, var_type, prefix, seed=1234):
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
            dict[var_name] = tf.Variable(tf.random_normal(var_shape, seed=seed))
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


# to write weight, bias dict of matrices into a text file 
def save_vardict_to_file(filebasename_prefix, vardict, epoch, seed=1234, dict_name="weight", pickling=False, sep="\t"):
    now = datetime.datetime.now()
    
    if dict_name == "bias":
        suffix = "es"
    elif dict_name == "weight":
        suffix = "s"
    else:
        raise Exception('Invalid dict_name.')
    
    filebasename = filebasename_prefix + "_ep" + str(epoch) + "_sd" + str(seed) + "_" + dict_name + suffix + "_" + str(now.isoformat())
    all_weights_str = ""

    # check for pickling
    if pickling:
        pickle_file = open(filebasename + ".pkl", "wb")
        pickle.dump(vardict, pickle_file)

    keys_ordered = sorted(vardict.keys(), reverse=True)
    for key in keys_ordered:
        layer_weights_str = ""
        weights = vardict[key]

        if(dict_name == "bias"):
            layer_weights_str = ""
            for ind in range(weights.shape[0]):
                layer_weights_str = layer_weights_str + str(weights[ind])
                
                if ind != (weights.shape[0]-1):
                    layer_weights_str = layer_weights_str + sep
                else:
                    layer_weights_str = layer_weights_str + "\n"
        elif(dict_name == "weight"):
            for row in range(weights.shape[0]):
                row_str = ""
                for col in range(weights.shape[1]):
                    row_str = row_str + str(weights[row, col])
                
                    if col != (weights.shape[1]-1):
                        row_str = row_str + sep
    
                layer_weights_str = layer_weights_str + row_str + "\n"
        
        all_weights_str = all_weights_str + layer_weights_str
    
    output_file = open(filebasename + ".txt", "w+")
    output_file.write(all_weights_str)
    output_file.close()
    output_file.close()

def calculate_indiv_function_values(matrix, scoring_func, axis=1):
    if('abs_' in scoring_func):
       matrix = np.abs(matrix)
    
    increasing_flag = False
    if('sum' in scoring_func):
        feature_scores = np.sum(matrix, axis=1)
    elif('avg' in scoring_func):
        feature_scores = np.mean(matrix, axis=1)
    elif('median' in scoring_func):
        feature_scores = np.median(matrix, axis=1)
    elif('min' in scoring_func):
        feature_scores = np.min(matrix, axis=1)
    elif('std' in scoring_func):
        feature_scores = np.std(matrix, axis=1)
        increasing_flag = True
    elif('max' in scoring_func):
        feature_scores = np.max(matrix, axis=1)
    elif('skew' in scoring_func):
        feature_scores = stats.skew(matrix, axis=1)
        increasing_flag = True
    elif('kurt' in scoring_func):
        feature_scores = stats.kurtosis(matrix, axis=1)
    else:
        raise Exception('Invalid scoring function: ' +str(scoring_func))

    return feature_scores, increasing_flag

# weighted mixture of two functions separated by a hyphen: e.g. skew-kurt
# individual scores are minmax-scaled to balance their contribution to composite scores
def weighted_mixture(matrix, scoring_func='skew-kurt', axis=1, weight1=0.5, scaling=True):
    weight2 = 1 - weight1

    scoring_funcs = scoring_func.split('-')

    feature_scores1, increasing_flag1 = calculate_indiv_function_values(matrix, scoring_func=scoring_funcs[0], axis=axis)
    feature_scores1 = (-feature_scores1) if increasing_flag1 else feature_scores1
        
    feature_scores2, increasing_flag2 = calculate_indiv_function_values(matrix, scoring_func=scoring_funcs[1], axis=axis)
    feature_scores2 = (-feature_scores2) if increasing_flag2 else feature_scores2

    if scaling:
        feature_scores1 = minmax_scale(feature_scores1)
        feature_scores2 = minmax_scale(feature_scores2)
        
    weighted_feature_scores = (weight1 * feature_scores1) + (weight2 * feature_scores2)

    return weighted_feature_scores, False

# returns a sorted list (decreasing order) of features per the given selection function
def sort_features(weights, scoring_func='sum', weight1=0.5, axis=1, scaling=True): # start here
    w1 = weights['w1'] # rows correspond to source neurons, columns to destination ones
    
    if('-' in scoring_func):
        feature_scores, increasing_flag = weighted_mixture(w1, scoring_func=scoring_func, axis=axis, weight1=weight1, scaling=True)
    else:
        feature_scores, increasing_flag = calculate_indiv_function_values(w1, scoring_func=scoring_func, axis=axis)

    feature_scores = (-feature_scores) if increasing_flag else feature_scores # if increasing order is desired, swap signs to flip order so that decreasing order sorting below is returned as desired
    sorted_features = np.argsort(feature_scores)[::-1]

    return sorted_features
