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
from cmath import nan

# get data using read_dataset()
# feed train_model to fmin()

# trains a neural network and returns loss value
# the function to be fed to hyperopt's fmin()
# X and Y are tf.placeholders, D is a Dataset object
def train_model(space):
    dataset_name = 'AML_3' #'diabetes_SMOTE' #'TCGA_survival_KIPAN'  #'AML_4' #'moons' #'psychencode' #'mnist'
    
    classification = True
    if 'TCGA_survival' in dataset_name:
        classification = False
    
    print("Dataset name: {0}".format(dataset_name))

    batch_size=int(space["batch_size"])
    learning_rate=float(space["learning_rate"])
    regularization_beta = float(space["regularization_beta"])
    n_hidden_1=int(space["n_hidden_1"])
    n_hidden_2=int(space["n_hidden_2"])
    activ_func1= str(space["activ_func1"])
    activ_func2= str(space["activ_func2"])
    activ_func3= str(space["activ_func3"])

    D = hf.read_dataset(dataset_name=dataset_name, one_hot_encoding=classification)
    X_tr, Y_tr = D.train.points, D.train.labels
    X_val, Y_val = D.validation.points, D.validation.labels
    X_ts, Y_ts = D.test.points, D.test.labels

    training_epochs = 3000
    display_step = min(250, int(training_epochs/2))

    n_input = D.train.points.shape[1] # number of features
    n_batch = int(D.train.points.shape[0]/batch_size)
    if 'TCGA_survival' in dataset_name:
        n_output = 3 # tensor cox output: start, end and event
        n_hidden_3 = int(space["n_hidden_3"])
        layer_sizes = [n_input, n_hidden_1, n_hidden_2, n_hidden_3]
    else:
        if classification:
            print(collections.Counter(np.argmax(Y_tr, 1).flatten()))
            n_output = len(np.unique(np.argmax(D.train.labels, axis=1)))
        else:
            n_output = 1
            
        layer_sizes = [n_input, n_hidden_1, n_hidden_2, n_output]

    print("Batch size: {0} \nlearning rate: {1}\nregularization_beta: {2} \nlayer_sizes: {3} \n" \
          "activ_func1: {4} \nactiv_func2: {5} \nactiv_func3: {6}".format(batch_size, 
          learning_rate, regularization_beta, layer_sizes, activ_func1, activ_func2, activ_func3))
    
    print("N_input: {0} \nn_output: {1} \nN_batch: {2} \n".format(n_input, n_output, n_batch))

    # Input data placeholder
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    layer_types = ["ff", "ff", "ff", "ff"]
    activ_funcs = [activ_func1, activ_func2, activ_func3]
    
    init_reduction = 'fan_in'
    fanin_std_devs = []
    if init_reduction == 'fan_in':
        fanin_std_devs = hf.get_fanin_stds(X_tr, layer_sizes, activ_funcs, selected_range=(-3, 3))
    
    weights = hf.get_vardict(layer_sizes, 'norm', 'weight', 'w', init_reduction, fanin_std_devs)
    biases = hf.get_vardict(layer_sizes, 'norm', 'bias', 'b', init_reduction, fanin_std_devs)

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
    
    # L2 regularization
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
                
            if(epoch % display_step == 0 or epoch == training_epochs): 
                print("Epoch: {0}".format(epoch))
                
                # Test model
                if 'TCGA_survival' not in dataset_name:
                    if classification:
                        pred = tf.nn.softmax(nnet[output_layer_index])  # Apply softmax to outputs
                    else:
                        pred = nnet[output_layer_index]
                    
                    tr_predictions = sess.run(pred, feed_dict={X: X_tr, Y: Y_tr})
                    val_predictions = sess.run(pred, feed_dict={X: X_val, Y: Y_val})
                
                if classification:
                    tr_accuracy = accuracy_score(np.argmax(Y_tr, 1), np.argmax(tr_predictions, 1))
                    val_accuracy = accuracy_score(np.argmax(Y_val, 1), np.argmax(val_predictions, 1))
                    print("Training Accuracy: {0}, Val. Accuracy: {1}".format(tr_accuracy, val_accuracy))
                    
                    if(n_output == 2):
                        tr_precision = precision_score(np.argmax(Y_tr, 1), np.argmax(tr_predictions, 1))
                        val_precision = precision_score(np.argmax(Y_val, 1), np.argmax(val_predictions, 1))
                        print("Training Precision: {0}, Val. Precision: {1}".format(tr_precision, val_precision))
                        
                        tr_auc = roc_auc_score(np.argmax(Y_tr, 1), np.argmax(tr_predictions, 1))
                        val_auc = roc_auc_score(np.argmax(Y_val, 1), np.argmax(val_predictions, 1))
                        print("Training AUC ROC: {0}, Val. AUC ROC: {1}".format(tr_auc, val_auc))
                        
                    if abs(tr_accuracy - val_accuracy) > 0.25:
                        print('Early termination; Tr-Val accuracy difference > 25%.')
                        break
                else:
                    if 'TCGA_survival' in dataset_name:
                        tr_concordance = sess.run(ci, feed_dict={X: X_tr, Y: Y_tr})
                        val_concordance = sess.run(ci, feed_dict={X: X_val, Y: Y_val})
                        print("Training CI: {0}, Val. CI: {1}".format(tr_concordance[0], val_concordance[0]))

                        if np.isnan(tr_concordance[0]):
                            print('Oops!')
                            break
                    else:
                        tr_mse = (np.square(Y_tr - tr_predictions).mean())
                        val_mse = (np.square(Y_val - val_predictions).mean())
                        
                        tr_r2 = r2_score(Y_tr, tr_predictions)
                        val_r2 = r2_score(Y_val, val_predictions)
                        print("Training MSE: {0}, Val. MSE: {1}\nTraining R2: {2}, Val. R2: {3}".format(tr_mse, val_mse, tr_r2, val_r2))
                
                end = time.time()
                print("Loss: {0}".format(loss_value))
                print("Epoch duration: {0}".format(str(end-start)+" sec.\n"))

        if 'TCGA_survival' in dataset_name:
            ts_concordance = sess.run(ci, feed_dict={X: X_ts, Y: Y_ts})
            print("Test CI: {0}".format(ts_concordance[0]))
            
            perf_score = hf.get_performance_score(tr_concordance[0], ts_concordance[0])
            return {'loss': -perf_score, 'perf_score': perf_score, 'tr_concordance': tr_concordance[0], 'ts_concordance': ts_concordance[0], 'val_accuracy': val_concordance[0], 'avg_loss': avg_loss_value, 'status': STATUS_OK}
        else:
            if classification:
                ts_predictions = sess.run(pred, feed_dict={X: X_ts, Y: Y_ts})
            
                ts_accuracy = accuracy_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
                print("Test Accuracy: {0}".format(ts_accuracy))
                
                if(n_output == 2):
                    ts_precision = precision_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
                    print("Test Precision: {0}".format(ts_precision))
                    
                    ts_auc = roc_auc_score(np.argmax(Y_ts, 1), np.argmax(ts_predictions, 1))
                    print("Test AUC ROC: {0}".format(ts_auc))
                    
                    perf_score = hf.get_performance_score(tr_auc, ts_auc)
                    return {'loss': -perf_score, 'perf_score': perf_score, 'tr_auc': tr_auc, 'ts_auc': ts_auc, 'val_auc': val_auc, 'tr_accuracy': tr_accuracy, 'ts_accuracy': ts_accuracy, 'val_accuracy': val_accuracy, 'tr_precision': tr_precision, 'ts_precision': ts_precision, 'val_precision': val_precision, 'avg_loss': avg_loss_value, 'status': STATUS_OK}
                else:
                    perf_score = hf.get_performance_score(tr_accuracy, ts_accuracy)
                    return {'loss': -perf_score, 'perf_score': perf_score, 'tr_accuracy': tr_accuracy, 'ts_accuracy': ts_accuracy, 'val_accuracy': val_accuracy, 'avg_loss': avg_loss_value, 'status': STATUS_OK}
            else:
                return {'mse': val_mse, 'loss': avg_loss_value, 'status': STATUS_OK}

    
def main():
    print("Start")
    hf.set_seed(1234)

    space = {   'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
                'regularization_beta': hp.uniform('regularization_beta', 0.001, 0.01), 
                'batch_size': hp.choice('batch_size', (128, 256, 512)),
                'activ_func1': hp.choice('activ_func1', ('relu', 'sigmoid')),
                'activ_func2': hp.choice('activ_func2', ('relu', 'sigmoid')),
                'activ_func3': hp.choice('activ_func3', ['linear']),
                'n_hidden_1': hp.uniform('n_hidden_1', 400, 600),
                'n_hidden_2': hp.uniform('n_hidden_2', 400, 600)
    }

    t = Trials()
    best = fmin(train_model, space=space, algo=tpe.suggest, max_evals=100, trials=t)
    print('TPE best: {}'.format(space_eval(space, best)))

    for trial in t.trials:
        try:
            trial_hyperparam_space = hf.unpack_dict(trial['misc']['vals'])        
            print('{} --> {}'.format(trial['result'], space_eval(space, trial_hyperparam_space)))
        except:
            print('Error with a hyperparameter space occurred.')
            continue

    print("Best results: {}".format(hf.get_best_result(t, space, metric='loss', direction='min')))

main()
