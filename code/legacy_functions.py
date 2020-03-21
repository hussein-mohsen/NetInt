# a helper function for sigmoid (and sigmoid-like functions), f, whose f(0) != 0
# these layers' outgoing weights must be reset to 0 before each batch update to ensure tuning runs accurately
# if layer wi is a sigmoid layer (i.e. activ_funcs[wi-2]='sigmoid' as activ_funcs' index starts at 0 to described layer 2), first hidden layer, outgoing off indices (i.e. rows) in weight matrix w_wi+1 are reset to 0
def tune_weights_before_batch_optimization(off_indices, weights, activ_funcs, tuning_layer_start, tuning_layer_end, sess):
    tuning_weight_start = max(1, tuning_layer_start-1)
    tuning_weight_end = tuning_layer_end

    for wi in range(tuning_weight_start, tuning_weight_end+1):
        if wi >= 2 and activ_funcs[wi-2] not in ('relu', 'linear'):
            tune_off_outgoing_weights(off_indices, weights, wi, sess)
            
# scale all values to [0,1]
def minmax_scale(values):
    values = (values - values.min())/((values.max() - values.min() + epsilon))
    return values

# scales empirical matrix to [0,1] per                                                                                        
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

##? might keep? incoming and outcoming weight-based distance values
# calculates KL divergence from a target distribution for incoming and outcoming weight distributions
# KL calculated after min-max scaling to [0, 1] + eps; returns averaged incoming and outcoming scores for each neuron
def calculate_layer_distance_values_old(weights_npdict, layer_index, 
                                    shift_type='min', scale_type='minmax', 
                                    target_distribution='norm', tuning_type='kl_div', 
                                    ks_metric='D', percentiles=False):    
    if layer_index > len(weights_npdict): # output layer or erroneous index
        raise Exception('Layer index is out of bounds.')
    else:
        outcoming_weights = weights_npdict['w'+str(layer_index)]        
        distance_values = calculate_distance_values(outcoming_weights, tuning_type=tuning_type,shift_type=shift_type, 
                                                    scale_type=scale_type, target_distribution=target_distribution, 
                                                    percentiles=percentiles, axis=1)

    
        if layer_index > 1: # beyond input layer
            incoming_weights = weights_npdict['w'+str(layer_index-1)]
            incoming_distance_values = calculate_distance_values(incoming_weights, tuning_type=tuning_type, shift_type=shift_type,
                                                                 scale_type=scale_type, target_distribution=target_distribution, 
                                                                 percentiles=percentiles, axis=0)

            distance_values = (distance_values + incoming_distance_values) / 2
            
    return distance_values

##? might keep? weight-based getting off_indices; includes centrality implementation
# gets indices to be tuned (i.e. turned off and replaced by 0 values)
# available indices are the ones not tuned prior to selection   
# tuning_type: 'centrality' (default: betweenness centrality) 
#              'kl_div' (default: with Gaussian)   
#              'random': 
def get_off_inds_old(weights_npdict, avail_inds, off_inds, layer_index, input_list=[],
                     k_selected=4, tuning_type='centrality', dt=[('weight', float)],
                     shift_type='min', scale_type='minmax', target_distribution='norm',
                     percentiles=False):
    if(len(avail_inds) == 0):
        select_inds = [-1]
        print('Warning: no more neurons to tune.')
    else:
        if tuning_type == 'random': # random selection of indices
            select_inds = random.sample(range(len(avail_inds)), k_selected) # indices within avail_inds to be turned off        
        else:
            if tuning_type == 'centrality': # sorted centrality-based selection
                increasing_flag = True # for centrality, higher neuron value is better; for distribution-based distance measures, lower is better.
                
                weight_graph, layer_boundaries = create_weight_graph(weights_npdict, layer_index)
                weight_graph = weight_graph.astype(dt)
                weight_G = nx.from_numpy_matrix(weight_graph) # create graph object
                
                # calculate centrality measure values
                print('Calculating centrality measures...')
                values = np.array(list(nx.betweenness_centrality(weight_G, k=7, weight='weight').values()))
                
                # select the nodes corresponding to the layer since the graph = its nodes + those of preceding 
                # and following layers
                layer_start = layer_boundaries[0]; layer_end = layer_boundaries[1]
                values = values[layer_start:layer_end]
            elif tuning_type == 'kl_div' or tuning_type == 'ks_test':
                increasing_flag = False
                
                # calculate KL divergence from a target distribution (default: Gaussian)
                print('Calculating distribution distance values per {0}...'.format(tuning_type))
                values = calculate_layer_distance_values(weights_npdict, layer_index, tuning_type=tuning_type,
                                                         shift_type=shift_type, scale_type=scale_type, 
                                                         target_distribution=target_distribution,
                                                         percentiles=percentiles)
            else:
                raise Exception('Invalid tuning type value.')
    
            # select k_selected nodes to tune
            inds = np.argsort(values)
            if increasing_flag == False:
                inds = inds[::-1]
                
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
# Layer indexing starts at 1 (input layer = 1, 1st hidden layer = 2, and so forth)
def create_weight_graph(weights_npdict, layer):
    if layer == 1: # input layer: edge case where weight graph is made of one layer
        return pad_matrix(weights_npdict['w'+str(layer)])
    elif layer > len(weights_npdict): # output layer or erroneous index
        raise Exception('Layer index is out of bounds.')
    else: # hidden layer: preceding, current and next layer forming the graph 
        boundaries = []
        total_n = 0
        
        # create boundaries matrix to help slicing the weight graph
        for l in range(layer-1, layer+1):
                total_n += weights_npdict['w'+str(l)].shape[0]
                boundaries.append(total_n)
        
        total_n += weights_npdict['w'+str(l)].shape[1]
        boundaries.append(total_n)
            
        # create the graph of combined nodes implemented as an undirected graph
        # hence each weight matrix is added twice (as is and as transpose)
        weight_graph = np.zeros((total_n, total_n))
        for l in range(layer-1, layer+2):

            weight_matrix = weights_npdict['w'+str(layer)]
            pre_weight_matrix = weights_npdict['w'+str(layer-1)]
            
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