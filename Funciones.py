# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 22:42:12 2025
Funciones adicionales
@author: paco2
"""
import numpy as np

# predict digit using image passed in
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    #print(index)
    return str(index)

# threshold slider handler
threshold = 100
def on_threshold(x):
    global threshold
    threshold = x