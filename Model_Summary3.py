# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:35:08 2021

@author: Mihai
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

pb_model_dir = "C:/Users/Mihai/dizertatie/model3.h5"

# Loading the Tensorflow Saved Model (PB)
model = tf.keras.models.load_model(pb_model_dir)
print(model.summary())


