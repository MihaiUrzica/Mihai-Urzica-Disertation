# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:56:13 2021

@author: Mihai
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

pb_model_dir = "C:/Users/Mihai/dizertatie/Model3/model3.h5'"
h5_model = "C:/Users/Mihai/dizertatie/Model3/model3.h5"

# Loading the Tensorflow Saved Model (PB)
model = tf.keras.models.load_model(pb_model_dir)
print(model.summary())

model.eval(model)