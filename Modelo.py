# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:20:35 2025

@author: paco2
"""
#Imports
import tensorflow as tf

# get mnist data and cache it
def get_mnist_data():
    # get mnist data 
    path = 'mnist.npz'

    # get data - this will be cached
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    return (x_train, y_train, x_test, y_test)

# train model with mnist data
def train_model(x_train, y_train, x_test, y_test):
    # set up TF model and train
    # callback 
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(logs)
            if(logs.get('accuracy') > 0.1):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()
    # normalise
    x_train, x_test = x_train/255.0, x_test/255.0
    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    print(model.summary())
    # fit model
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    # stats
    print(history.epoch, history.history['accuracy'][-1])
    return model
