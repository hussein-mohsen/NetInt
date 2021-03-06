import helper_functions as hf
import tensorflow as tf
import tensorcox as tx
from hyperopt import fmin, hp, Trials, tpe, space_eval, STATUS_OK

import numpy as np
import time

from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, r2_score

import collections
import random
import sys

# get data using read_dataset()
# feed train_model to fmin()

# trains a neural network and returns loss value
# the function to be fed to hyperopt's fmin()
# X and Y are tf.placeholders, D is a Dataset object
def train_model(space):
    dataset_name = 'TCGA_survival_KIPAN'  #'AML_4' #'moons' #'psychencode' #'mnist'
    classification = False
    scaling_type = 'standard'
    
    print("Dataset name: {0}".format(dataset_name))

    batch_size=int(space["batch_size"])
    learning_rate=float(space["learning_rate"])
    n_hidden_1=int(space["n_hidden_1"])
    n_hidden_2=int(space["n_hidden_2"])
    activ_func1= str(space["activ_func1"])
    activ_func2= str(space["activ_func2"])
    activ_func3= str(space["activ_func3"])

    D = hf.read_dataset(dataset_name=dataset_name, one_hot_encoding=classification, scaling_type=scaling_type)
    X_tr, Y_tr = D.train.points, D.train.labels
    X_ts, Y_ts = D.test.points, D.test.labels

    training_epochs = 800
    display_step = min(50, int(training_epochs/2))

    n_input = D.train.points.shape[1] # number of features
    n_batch = int(D.train.points.shape[0]/batch_size)
    if 'TCGA_survival' in dataset_name:
        n_output = 3 # tensor cox output: start, end and event
        n_hidden_3 = int(space["n_hidden_3"])
        layer_sizes = [n_input, n_hidden_1, n_hidden_2, n_hidden_3]
    else:
        if classification:
            print(collections.Counter(np.argmax(Y_ts, 1).flatten()))
            n_output = len(np.unique(np.argmax(D.train.labels, axis=1)))
        else:
            n_output = 1
            
        layer_sizes = [n_input, n_hidden_1, n_hidden_2, n_output]

    print("Batch size: {0} \nlearning rate: {1} \nlayer_sizes: {2} \n" \
          "activ_func1: {3} \nactiv_func2: {4} \nactiv_func3: {5}".format(batch_size, 
          learning_rate, layer_sizes, activ_func1, activ_func2, activ_func3))
    
    print("N_input: {0} \nn_output: {1} \nN_batch: {2} \n".format(n_input, n_output, n_batch))

    # Input data placeholder
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    layer_types = ["ff", "ff", "ff", "ff"]
    activ_funcs = [activ_func1, activ_func2, activ_func3]
    
    init_reduction = 'fan_in'
    weights = hf.get_vardict(layer_sizes, 'norm', 'weight', 'w', activ_funcs, init_reduction)
    biases = hf.get_vardict(layer_sizes, 'norm', 'bias', 'b', activ_funcs, init_reduction)

    print("Layer sizes: {}".format(layer_sizes)) 
    print("Activation_funcs: {}".format(activ_funcs))

    # get MLP
    nnet = hf.multilayer_perceptron(X, weights, biases, activ_funcs, layer_types)
    output_layer_index = len(layer_sizes)-1 # since indexing starts from 0
    
    if classification: # discrete
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nnet[output_layer_index], labels=Y))
    else:
        if 'TCGA_survival' in dataset_name:
            #surv_ = tf.placeholder(tf.float32, shape=[None, 3])
            theta_ = tf.Variable(initial_value=tf.random_normal([int(nnet[output_layer_index].shape[1]), 1], mean=0, stddev=0.1, dtype=tf.float32))
            pred_ = tf.matmul(nnet[output_layer_index], theta_)
            
            # Tensorcox
            tcox = tx.tensorcox(tf.cast(Y, tf.float64), tf.cast(pred_, tf.float64))
            loss_op = -tcox.loglikelihood()
            ci = tcox.concordance()
        else: # regression
            loss_op = tf.losses.mean_squared_error(predictions=nnet[output_layer_index], labels=Y)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
        
    # Initialize the variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
                
        for epoch in range(1, training_epochs+1):        
            avg_loss_value = 0.0
            
            start = time.time()
            
            # Loop over all batches
            for i in range(n_batch):
                batch_x, batch_y = D.train.next_batch(batch_size)
                
                _, loss_value = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

                # Compute average loss
                avg_loss_value += loss_value / n_batch
                    
            end = time.time()

            if(epoch % display_step == 0): 
                # Test model
                if 'TCGA_survival' not in dataset_name:
                    if classification:
                        pred = tf.nn.softmax(nnet[output_layer_index])  # Apply softmax to outputs
                    else:
                        pred = nnet[output_layer_index]
                    
                    ts_predictions = sess.run(pred, feed_dict={X: X_ts, Y: Y_ts})
                
                if classification:
                    accuracy = accuracy_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
                    print("Test Accuracy: {0}".format(accuracy))
                    
                    if(n_output == 2):
                        precision = precision_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
                        print("Test Precision: {0}".format(precision))
                        
                        auc = roc_auc_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
                        print("Test AUC ROC: {0}".format(auc))
                else:
                    if 'TCGA_survival' in dataset_name:
                        tr_concordance = sess.run(ci, feed_dict={Y: Y_tr, X: X_tr})
                        concordance = sess.run(ci, feed_dict={Y: Y_ts, X: X_ts})
                        print("Training CI: {0}, Test CI: {1}".format(tr_concordance, concordance))
                    else:
                        mse = (np.square(Y_ts - ts_predictions).mean())
                        r2 = r2_score(Y_ts, ts_predictions)
                        print("Test MSE: {0}, Test R2: {1}".format(mse, r2))

                print("Epoch: {0}".format(epoch))
                print("Loss: {0}".format(loss_value))
                print("Epoch duration: {0}".format(str(end-start)+" sec.\n"))
                
    # train and return loss
    if 'TCGA_survival' in dataset_name:
        return {'concordance': concordance, 'loss': avg_loss_value, 'status': STATUS_OK}
    else:    
        if classification:
            if n_output == 2:
                return {'auc': auc, 'accuracy': accuracy, 'precision': precision, 'loss': avg_loss_value, 'status': STATUS_OK}
            else:
                return {'accuracy': accuracy, 'loss': avg_loss_value, 'status': STATUS_OK}
        else:
            return {'mse': mse, 'loss': avg_loss_value, 'status': STATUS_OK}

    
def main():
    print("Start")
    hf.set_seed(1234)

    '''
    # to train a single model
    space = {
        'learning_rate': 0.015959604297247607,
        'batch_size': 247,
        'activ_func1': 'sigmoid',
        'activ_func2': 'sigmoid',
        'activ_func3': 'softmax',
        'n_hidden_1': 6,
        'n_hidden_2': 104
    }
        
    trainings_results = train_model(space)
    print(trainings_results)
    '''

    # expanded space
    space = {
                'learning_rate': hp.uniform('learning_rate', 0.000001, 0.0001), 
                'batch_size': hp.choice('batch_size', (8, 16, 32)), 
                #'batch_size': hp.choice('batch_size', (128, 256, 512, 1024)), 
                'activ_func1': hp.choice('activ_func1', ('relu', 'sigmoid')), 
                'activ_func2': hp.choice('activ_func2', ('relu', 'sigmoid')), 
                'activ_func3': hp.choice('activ_func3', ['linear']), 
                'n_hidden_1': hp.uniform('n_hidden_1', 2000, 3000), 
                'n_hidden_2': hp.uniform('n_hidden_2', 2000, 3000),
                'n_hidden_3': hp.uniform('n_hidden_3', 2000, 3000) # used only for cox models
    }

    t = Trials()
    best = fmin(train_model, space=space, algo=tpe.suggest, max_evals=50, trials=t)
    print('TPE best: {}'.format(space_eval(space, best)))

    for trial in t.trials:
        try:
            trial_hyperparam_space = hf.unpack_dict(trial['misc']['vals'])        
            print('{} --> {}'.format(trial['result'], space_eval(space, trial_hyperparam_space)))
        except:
            print('Error with a hyperparameter space occurred.')
            continue

    print("Best results: {}".format(hf.get_best_result(t, space, metric='accuracy')))

main()
