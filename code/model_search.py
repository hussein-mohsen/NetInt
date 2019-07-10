from helper_functions import read_dataset, multilayer_perceptron, get_vardict, unpack_dict

from hyperopt import fmin, hp, Trials, tpe, space_eval, STATUS_OK
import tensorflow as tf

import numpy as np
import time

from sklearn.preprocessing import normalize
from IPython.core.display import display

# get data using read_dataset()
# feed train_model to fmin()

# trains a neural network and returns loss value
# the function to be fed to hyperopt's fmin()
# X and Y are tf.placeholders, D is a Dataset object
def train_model(space):
    dataset_name = 'mnist' #'diabetes' #'psychencode' #'mnist'
    D = read_dataset(dataset_name=dataset_name, minmax_scaling=True)
    X_tr, Y_tr = D.train.points, D.train.labels
    X_ts, Y_ts = D.test.points, D.test.labels

    batch_size=int(space["batch_size"])
    learning_rate=float(space["learning_rate"])
    n_hidden_1=int(space["n_hidden_1"])
    n_hidden_2=int(space["n_hidden_2"])
    activ_func1= str(space["activ_func1"])
    activ_func2= str(space["activ_func2"])
    activ_func3= str(space["activ_func3"])
    
    training_epochs = 300
    display_step = min(50, int(training_epochs/2))
        
    print("Batch size: {0} \nlearning rate: {1} \nn_hidden_1: {2} \nn_hidden_2: {3} \n" \
          "activ_func1: {4} \nactiv_func2: {5} \nactiv_func3: {6}".format(batch_size, 
          learning_rate, n_hidden_1, n_hidden_2, activ_func1, activ_func2, activ_func3))

    n_input = D.train.points.shape[1] # number of features
    n_classes = len(np.unique(np.argmax(D.train.labels, axis=1)))
    n_batch = int(D.train.points.shape[0]/batch_size)

    print("N_input: {0} \nN_classes: {1} \nN_batch: {2} \n".format(n_input, n_classes, n_batch))

    # Input data placeholder
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    layer_sizes = [n_input, n_hidden_1, n_hidden_2, n_classes]
        
    layer_types = ["ff", "ff", "ff", "ff"]
    activ_funcs = [activ_func1, activ_func2, activ_func3]
    
    weights = get_vardict(layer_sizes, 'norm', 'weight', 'w')
    biases = get_vardict(layer_sizes, 'norm', 'bias', 'b')

    print("Layer sizes: {}".format(layer_sizes)) 
    print("Activation_funcs: {}".format(activ_funcs))

    # get MLP
    logits = multilayer_perceptron(X, weights, biases, activ_funcs, layer_types)
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
        
    # Initialize the variables

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1, training_epochs):
            avg_loss_value = 0.0
            
            start = time.time()
            
            # Loop over all batches
            for i in range(n_batch):
                batch_x, batch_y = D.train.next_batch(batch_size)
                #normalize(batch_x, axis=0, norm='max')
                
                _, loss_value = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        
                # Compute average loss
                avg_loss_value += loss_value / n_batch
                    
            end = time.time()

            if(epoch % display_step == 0):
                # Test model
                # Calculate training accuracy        
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)) 
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                                
                # Calculate test accuracy
                accuracy_value = accuracy.eval({X: X_ts, Y: Y_ts})
            
                print('Epoch: '+str(epoch))
                print("Accuracy:"+str(accuracy_value))
                print("Epoch duration: "+str(end-start)+" sec.\n")
                
    # train and return loss
    return {'accuracy': accuracy_value, 'loss': avg_loss_value, 'status': STATUS_OK}

    
def main():
    print("Start")

    '''
    # to train a single model
    space = {
        'learning_rate': 0.01,
        'batch_size': 200,
        'activ_func1': 'relu',
        'activ_func2': 'relu',
        'activ_func3': 'softmax',
        'n_hidden_1': 1200,
        'n_hidden_2': 1200
    }
        
    train_model(space)
    '''
    
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.05),
        'batch_size': hp.uniform('batch_size', 50, 250),
        'activ_func1': hp.choice('activ_func1', ('relu', 'sigmoid')),
        'activ_func2': hp.choice('activ_func2', ('relu', 'sigmoid')),
        'activ_func3': hp.choice('activ_func3', ['softmax']),
        'n_hidden_1': hp.uniform('n_hidden_1', 500, 1200),
        'n_hidden_2': hp.uniform('n_hidden_2', 500, 1200)
    }
    
    t = Trials()
    best = fmin(train_model, space=space, algo=tpe.suggest, max_evals=3, trials=t)
    print('TPE best: {}'.format(space_eval(space, best)))

    for trial in t.trials:
        try:
            trial_hyperparam_space = unpack_dict(trial['misc']['vals'])        
            print('{} --> {}'.format(trial['result'], space_eval(space, trial_hyperparam_space)))
        except:
            print('Error with a hyperparameter space occurred.')
            continue
    
main()
