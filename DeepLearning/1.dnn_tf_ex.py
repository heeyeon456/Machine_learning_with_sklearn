import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tools import rmse_cal,mae_cal,cor_cal,mean_cal,frange,\
                    accuracy,precision,recall,aupr,f1_score,make_binary
from validation import classification_cv,regression_cv,external_val_reg,\
                        external_val_classif, test_preprocessing, \
                        draw_roc,cal_external_auc

# load the data
enc = OneHotEncoder()

dataset=pd.read_table('../data/breast_cancer_svc.tsv',sep='\t')

input_data=dataset.iloc[:,1:].transpose()
X_data=input_data.iloc[:,:-1].values
y_data=input_data.iloc[:,-1]
y_data=make_binary('normal','cancer',y_data)

# preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.1, random_state=42)

y_train = enc.fit_transform(y_train.values.reshape(-1,1)).toarray()
y_test = enc.fit_transform(y_test.values.reshape(-1,1)).toarray()

# hyperparameters
# variable 정의
X_dim = X_train.shape[1]
input_n = X_train.shape[0]
hidden_1 = 200
hidden_2 = 100
output_size = 2
lr = 1e-04
max_epoch = 50
dropout_prob = 0.5

# create network
# create fully connected layer
def dense(input_layer,layer_size):
    
    init = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    
    # Batchnorm settings
    #training_phase = tf.placeholder(tf.bool, phase, name='training_phase')
    
    HiddenLayer = tf.layers.dense(inputs = input_layer, units = layer_size, 
                              activation=None, # Batchnorm comes before nonlinear activation
                              use_bias=False, # Note that no bias unit is used in batchnorm
                              kernel_initializer=init, kernel_regularizer = regularizer)
    
    HiddenLayer = tf.layers.batch_normalization(HiddenLayer)
    HiddenLayer = tf.nn.relu(HiddenLayer)
    
    return HiddenLayer

# define the network
tf.reset_default_graph()


# input 형태 정의
X = tf.placeholder(tf.float32, [None, X_dim])
y = tf.placeholder(tf.float32,[None,2])

with tf.name_scope("DNN"):
    
    keep_proba = tf.placeholder(tf.float32, None, name='keep_proba')

    
    hidden_layer1 = dense(X,hidden_1)
    hidden_layer2 = dense(hidden_layer1, hidden_2)
    hidden_layer2 = tf.nn.dropout(hidden_layer2, keep_prob=dropout_prob)
    output_layer = dense(hidden_layer2, output_size)
     
with tf.name_scope("loss"):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y)
    cost = tf.reduce_mean(loss, name = 'loss')

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    training = optimizer.minimize(loss,name='training')
    
with tf.name_scope("eval"):
    correct = tf.equal(tf.argmax(y,1), tf.argmax(output_layer,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='eval')

# run the session(training)
init = tf.global_variables_initializer()
epoch_count = 0
best_accu_valid = 0
with tf.Session() as sess:
    init.run()
    while epoch_count < max_epoch :
        
        sess.run(training,feed_dict={X:X_train,y:y_train})
        accu_train = accuracy.eval(feed_dict={X:X_train,y:y_train})
        accu_valid = accuracy.eval(feed_dict={X:X_test,y:y_test})
        
        if accu_valid > best_accu_valid:
            best_accu_valid = accu_valid
            
        epoch_count+=1
        
        print('Epoch : ',epoch_count, '| Training Accuracy:',accu_train,'  Validation Accuracy:',accu_valid)

