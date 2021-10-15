import os
import pickle

import helper_functions as hf
import numpy as np

from scipy import stats
from scipy.spatial import distance



# calculate scores of an individual function
def calculate_indiv_function_values(matrix, scoring_func, axis=1):
    if('abs_' in scoring_func):
       matrix = np.abs(matrix)
       scoring_func = scoring_func[4:]
    
    increasing_flag = False
    if('sum' in scoring_func):
        feature_scores = np.sum(matrix, axis=axis)
    elif('avg' in scoring_func):
        feature_scores = np.mean(matrix, axis=axis)
    elif('median' in scoring_func):
        feature_scores = np.median(matrix, axis=axis)
    elif('min' in scoring_func):
        feature_scores = np.min(matrix, axis=axis)
        increasing_flag = True
    elif('std' in scoring_func):
        feature_scores = np.std(matrix, axis=axis)
        increasing_flag = True
    elif('max' in scoring_func):
        feature_scores = np.max(matrix, axis=axis)
    elif('skew' in scoring_func):
        feature_scores = stats.skew(matrix, axis=axis)
        increasing_flag = True
    elif('kurt' in scoring_func):
        feature_scores = stats.kurtosis(matrix, axis=axis)
    elif('kl_div' in scoring_func or 'ks_test' in scoring_func): #e.g. kl_div:norm, ks_test:norm, etc.
        values = scoring_func.split(':')
        distance_measure = values[0]
        target_distribution = values[1]

        # Note: For weights outcoming from input matrix, axis = 1
        feature_scores = hf.calculate_distance_values(matrix, tuning_type=distance_measure, shift_type='min', scale_type='minmax', target_distribution=target_distribution, axis=axis)
        increasing_flag = True
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

    #feature_scores = (-feature_scores) if increasing_flag else feature_scores # if increasing order is desired, swap signs to flip order so that decreasing order sorting below is returned as desired
    sorted_features = np.argsort(feature_scores)
    sorted_features = sorted_features if increasing_flag else sorted_features[::-1]
    print(sorted_features)

    return sorted_features

# calculates x and y coordinates from a pixel's index in a flat vector
def calculate_2D_coordinates(index, n_cols):
    x = int(index / n_cols)
    y = int(index % n_cols)
    
    return [x, y]

# calculates 2D Euclidean distance between coordinates of two pixel indices
def calculate_2D_euclidean_distance(index1, index2, n_cols):
    return distance.euclidean(calculate_2D_coordinates(index1, n_cols=n_cols), calculate_2D_coordinates(index2, n_cols=n_cols))

def calculate_img_features_pvalue(sorted_input_features, input_data, top_k, n_cols, n_trials=10000):
    # calculate p-value based on euclidean distance between the points
    # calculate coordinates of the top_k input features selected by our algorithm
    
    # calculate coordinates of the most significant pixels in input dataset (i.e. ones with highest intensity) to be used as gold standard
    # calculate heatmap by stacking all images on top of each other: assumes each image is flattened to 1D
    heatmap = np.sum(input_data, axis=0)
    heatmap_sorted_features = np.argsort(heatmap)[::-1]
    top_k_features = heatmap_sorted_features[0:top_k]
    top_k_points = list(map(calculate_2D_coordinates, top_k_features, np.repeat(n_cols, repeats=top_k)))
    
    top_k_input_features = sorted_input_features[0:top_k]
    top_k_input_points = list(map(calculate_2D_coordinates, top_k_input_features, np.repeat(n_cols, repeats=top_k)))    
    top_k_input_feature_scores = np.min(distance.cdist(top_k_input_points, top_k_points), axis=1) # calculate the minimum distance each of the top input features has with one of the gold standard features
    instance_score = np.median(top_k_input_feature_scores) # calculate the median of the top_k scores as a collective instance score to use while calculating the pvalue

    #print(top_k_input_feature_scores)
    #print(instance_score)
    
    # H0: Selected features are not close top features
    # H1: ^H0
    n_features = input_data.shape[1]    
    
    n_lower_than_instance_score = 0.0
    for i in range(n_trials):
        randomly_selected_features = np.random.choice(n_features, top_k)
        randomly_selected_points = list(map(calculate_2D_coordinates, randomly_selected_features, np.repeat(n_cols, repeats=top_k)))
        randomly_selected_feature_scores = np.min(distance.cdist(randomly_selected_points, top_k_points), axis=1) # calculate the minimum distance each of the randomly selected features has with one of the gold standard features
        randomly_selected_instance_score = np.median(randomly_selected_feature_scores) # calculate the median of the top_k scores as a collective instance score to use while calculating the pvalue
        
        if randomly_selected_instance_score <= instance_score:
            n_lower_than_instance_score += 1

    p_value = n_lower_than_instance_score / n_trials

    return p_value

# evaluates selected features
# input_data/input_labels are data points to evaluate upon, if applicable (i.e. in mnist/image data visualizing input data points elucidates prioritized features)
# sorted_input_features are features to be assessed
# weigted_npdict is the dictionary of weight np ndarrays generated during training
# scoring functions is a list of scoring functions according to which evaluation will take place
# top_k is the number of features to be selected
# n_imgs is the number of images to be selected as a sample
# discarded_features are indices of features to be excluded from comparisons (e.g. could not be ranked)
# bottom_features is a boolean flag to determine if we need to assess bottom features (e.g. for diabetes dataset as opposed to xor where only top features matter)
# visualize_img is a parameter to select between assessment methods for image data (if True, visualization with selected pixels highlighted, otherwise, p_value generation per custom test above) 
# n_trials are the ones used to generate the empirical p_value in case visualize_img=False 
def evaluate_features(dataset_name, input_matrix, scoring_functions, eval_type, sorted_ref_features=[], discarded_features=[], output_dir='', uid=1234, top_k=10, input_data=[], input_labels=[], bottom_features=True, visualize_imgs=True, n_imgs=-1, n_trials=10000):        
    output_dir += str(uid)+'/'
    os.makedirs(output_dir)
    print('Results will be saved in ' + str(output_dir))
    
    results_txt = ''

    if eval_type == 'img' and visualize_imgs == False:
            results_filename = str(uid)+'_pvalues.txt'
            
    if n_imgs != -1: # randomly select n_imgs if parameter is not empty
        input_data = input_data[np.random.choice(input_data.shape[0], n_imgs, replace=False), :]
        print('Randomly selected ' +str(n_imgs)+ ' images.')

    if eval_type == 'rank':
        results_filename = str(uid)+'_selected_features.txt'

    for sf in scoring_functions:
        if sf not in ['', 'abs_']:
            sorted_input_features = sort_features(weights_npdict, scoring_func=sf)
            features_filename = output_dir+dataset_name+'_'+sf+'_sorted_features.pkl'
            pickle.dump(sorted_input_features, open(features_filename, 'wb'))
            print('Pickled ' +features_filename)
    
            if eval_type == 'img':
                if visualize_imgs: # if image data evaluation is through visualization
                    # reduce color of pixels to highlight selected features
                    sf_input_data = np.copy(input_data)
                    sf_input_data = sf_input_data / 1.5
                    sf_input_data[:, sorted_input_features[1:top_k]] = 1 # highlight selected features in the image
                    sf_input_data = sf_input_data.reshape(sf_input_data.shape[0], 28, 28) # reshape images
                    
                    sf_output_dir = output_dir +sf+'/'
                    os.makedirs(sf_output_dir)
                    for i in range(sf_input_data.shape[0]):
                        plt.imsave(sf_output_dir+'img'+str(i)+'.jpg', sf_input_data[i, :], cmap='viridis')
                        print('Image ' +str(i)+' saved.')
                else:
                    # calculate pvalue of selected features
                    sf_pval = calculate_img_features_pvalue(sorted_input_features, input_data, top_k=25, n_cols=28, n_trials=n_trials)
                    results_txt += (sf+ ': ' +str(sf_pval)+ '\n')
            else: # datasets with sorted input features (e.g. diabetes)
                diff = set(np.arange(sorted_input_features.shape[0])) - set(discarded_features)
                sorted_input_features = [o for o in sorted_input_features if o in diff]
    
                top_k_results = np.intersect1d(sorted_input_features[0:top_k], sorted_ref_features[0:top_k])
                results_txt += (sf+':\nTop '+str(top_k)+' results: ' +str(top_k_results) +'\n') 
                print(sf+':\nTop '+str(top_k)+' results: ' +str(top_k_results))
                print("\n")
                if bottom_features:
                    bottom_k_results = np.intersect1d(sorted_input_features[-top_k:], sorted_ref_features[-top_k:])
                    results_txt += ('Bottom '+str(top_k)+' results: ' +str(bottom_k_results) +'\n') 
                    print('Bottom '+str(top_k)+' results: ' +str(bottom_k_results))

    if len(results_txt) > 0:
        output_file = open(output_dir+'/'+results_filename, 'w') # write results to file
        output_file.write(results_txt)
        output_file.close()
