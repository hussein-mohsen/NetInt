from helper_functions import read_dataset, multilayer_perceptron, get_vardict

from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
import tensorflow as tf

import numpy as np
import time

# get data using read_dataset()
# feed train_model to fmin()

# trains a neural network and returns loss value
# the function to be fed to hyperopt's fmin()
# X and Y are tf.placeholders, D is a Dataset object
def train_model(space):
    dataset_name = 'diabetes'
    D = read_dataset(dataset_name)
    X_tr, Y_tr = D.train.points, D.train.labels
    X_ts, Y_ts = D.test.points, D.test.labels

    batch_size=int(space["batch_size"])
    learning_rate=space["learning_rate"]
    n_hidden_1=int(space["n_hidden_1"])
    n_hidden_2=int(space["n_hidden_2"])
    activ_func1="linear"#space["activ_func1"]
    activ_func2="linear"#space["activ_func2"]
    activ_func3="linear"#space["activ_func3"]
    
    #n_hidden_1=500
    #n_hidden_2=100
    regularization_rate=0.1
    
    training_epochs = 10
    
    print("Batch size: {}".format(batch_size))
    print("Training data shape: {}".format(D.train.points.shape))
    
    n_input = D.train.points.shape[1] # number of features
    n_classes = len(np.unique(np.argmax(D.train.labels, axis=1)))
    n_batch = int(D.train.points.shape[0]/batch_size)

    print("N_batch: {}".format(n_batch))
    print("N_classes: {}".format(n_classes))

    # Input data placeholder
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    layer_sizes = [n_input, n_hidden_1, n_hidden_2, n_classes]
    
    print("Layer sizes: {}".format(layer_sizes))
        
    weights = get_vardict(layer_sizes, 'norm', 'weight', 'w')
    biases = get_vardict(layer_sizes, 'norm', 'bias', 'b')

    layer_types = ["ff", "ff", "ff", "ff"]
    activ_funcs = [activ_func1, activ_func2, activ_func3]
    
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

                _, loss_value = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                
                
                ''' block added for debugging'''
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                pr = sess.run(pred, feed_dict={X:batch_x})
                #print(batch_y)
                #print("===")
                #print(pr)
        
                # Compute average loss
                avg_loss_value += loss_value / n_batch
                
            end = time.time()
            
            w1 = sess.run(weights['w1'], feed_dict={X: batch_x})
            print("Weights of Epoch ")
            print(w1)    
            end = time.time()
            
        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_value = accuracy.eval({X: X_ts, Y: Y_ts})
        
        print('Epochs: ', training_epochs)
        print("Accuracy:", accuracy_value)
    
        # train and return loss
        return {'loss': avg_loss_value, 'accuracy': accuracy_value, 'status': STATUS_OK}

def main():    
    space = {
        'learning_rate': 0.1,
        'batch_size': 200,
        #'activ_func1': hp.choice('activ_func1', ['sigmoid']),
        #'activ_func2': hp.choice('activ_func2', ['sigmoid']),
        #'activ_func3': hp.choice('activ_func3', ['sigmoid']),
        'n_hidden_1': 200,
        'n_hidden_2': 250
    }
        
    train_model(space)

    '''
    # read dataset
    dataset_name = 'diabetes'
    D = read_dataset(dataset_name)
    
    space = {
        'learning_rate': hp.loguniform('learning_rate', 0, 0.5),
        'batch_size': hp.quniform('batch_size', 20, 250, 25),
        #'activ_func1': hp.choice('activ_func1', ['sigmoid']),
        #'activ_func2': hp.choice('activ_func2', ['sigmoid']),
        #'activ_func3': hp.choice('activ_func3', ['sigmoid']),
        'n_hidden_1': hp.quniform('n_hidden_1', 50, 400, 10),
        'n_hidden_2': hp.quniform('n_hidden_2', 50, 400, 10)
    }

    t = Trials()
    best = fmin(train_model, space=space, algo=tpe.suggest, max_evals=100, trials=t)
    print('TPE best: {}'.format(best))

    for trial in t.trials:
        print('{} --> {}'.format(trial['result'], trial['misc']['vals']))
    '''
    
main()
