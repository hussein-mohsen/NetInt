import numpy as np
#from numpy import dtype, shape

import tensorflow as tf
import networkx as nx

from sklearn.preprocessing import minmax_scale

# calculates KL divergence of two distributions
def kl_div(empirical, target):
    if(abs(empirical.sum()-1) > 0.05 or abs(target.sum()-1) > 0.05):
        print("Warning: distributions do not sume up to 1")
        
    kl_div_value = (empirical * np.log(empirical/target)).sum()
    return kl_div_value

# scales a values s.t. sum = 1
def totality_scale(values):
    total = values.sum()
    return values/total

# scales empirical matrix to [0,1] + shift_factor and calculate KL div with Gaussian(mn=0, sd=0.1) + shift_factor
def calculate_scaled_kl_div(input_matrix, shift_factor=5, target_dist='Gaussian'):
    #input_matrix = minmax_scale(input_matrix, feature_range=(0,1), axis=1) +0.001 #+ shift_factor
    #target_dist = np.random.normal(1, 0.1, (1, input_matrix.shape[0])) +0.001 #+ shift_factor
    
    input_matrix += abs(input_matrix.min()) + 0.001
    input_matrix = np.apply_along_axis(totality_scale, 1, input_matrix)
    target_dist = totality_scale(np.random.normal(1, 0.1, (1, input_matrix.shape[1])))

    kl_values = np.apply_along_axis(kl_div, 1, input_matrix, target_dist)
    return kl_values

# calculates KL divergence from a target distribution for incoming and outcoming weight distributions
# KL calculated after min-max scaling to [0, 1] + eps; returns averaged incoming and outcoming scores for each neuron
def calculate_kl_div_layer_values(weights_dict, layer_index, target_dist='Gaussian'):    
    if layer_index > len(weights_dict): # output layer or erroneous index
        raise Exception('Layer is out of bounds.')
    else:
        outcoming_weights = weights_dict['w'+str(layer_index)]
        kl_values = calculate_scaled_kl_div(outcoming_weights)
        return kl_values
    
        if layer_index > 1:
            incoming_weights = weights_dict['w'+str(layer_index-1)]
            incoming_kl_values = calculate_scaled_kl_div(incoming_weights, axis=1)
            kl_values = (kl_values + incoming_kl_values) / 2
            
    return kl_values

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
# tuning_type: 'centrality' (default: betweenness centrality) 
#              'kl_div' (default: with Gaussian)   
#              'random': 
def get_off_inds(weights_dict, avail_inds, layer_index, input_list=[], k_selected=4, tuning_type='centrality', dt=[('weight', float)]):    
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
        elif tuning_type == 'kl_div':
            # calculate KL divergence from a target distribution (default: Gaussian)
            print("Calculating KL-divergence values...")
            values = calculate_kl_div_layer_values(weights_dict, layer_index, target_dist='Gaussian')

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

# helper function that turns off neurons at off_indices (0 value assignment)
# returns a tf tensor
def tune_weights(off_indices, current_weights, layer):
    current_weights['w'+str(layer)][off_indices, :] = 0 # turn off connections outcoming from off_indices neurons in the layer
    print("Outcoming connections at layer {} tuned.".format(layer))
     
    if(layer > 1):
        current_weights['w'+str(layer-1)][:, off_indices] = 0 # turn off connections incoming to off_indices neurons in the layer
        print("Incoming connections at layer {} tuned.".format(layer))
        
    return tf.convert_to_tensor(current_weights['w'+str(layer)], dtype=tf.float32)