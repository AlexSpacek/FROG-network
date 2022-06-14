import matplotlib.pyplot as plt
import numpy as np
import os
from my_classes import DataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from keras.layers import ( Conv2D, Flatten, Lambda, Dense, concatenate,
                         Dropout, Input )
from keras.models import Model
from sklearn.utils import class_weight
import sys


# Parameters
params = {'dim': (256,256),
          'batch_size': 25,
          'n_classes': 128,
          'n_channels': 1,
          'shuffle': True}
nTraining = 25000
nValidation = 5000
# Datasets
pTrain = np.linspace(1, nTraining, nTraining)
pValidate = np.linspace(nTraining+1, nTraining+nValidation, nValidation)
p2 = ["%gTrace" % x for x in pTrain]
p3 = ["%gTrace" % x for x in pValidate]
partition = {'train': p2, 'validation': p3}
lTrain = np.linspace(1, nTraining, nTraining)
lValidate = np.linspace(nTraining+1, nTraining+nValidation, nValidation)
l2 = ["%gLabel" % x for x in lTrain]
l3 = ["%gLabel" % x for x in lValidate]
labels = {'train': l2, 'validation': l3}


# Generators
training_generator = DataGenerator(partition['train'], labels['train'], **params)
validation_generator = DataGenerator(partition['validation'], labels['validation'], **params)

# Design model
from tensorflow.keras.applications import DenseNet121
from keras.activations import elu
custom_elu = lambda x: elu(x, alpha=0.01)
image_input = Input(shape=(256, 256, 3))
model_d = DenseNet121(weights='imagenet', include_top=False, pooling='max', classes=1024, input_shape=(256, 256, 3))
x = model_d.output
d2 = Dense(512, activation=custom_elu)(x)
out = Dense(128, activation=custom_elu)(d2)
model = Model(inputs=model_d.input, outputs=[out])
model.compile(loss='mse', optimizer='Adam',metrics=['mean_squared_error'])

# Train model on dataset
checkpoint_path = "Checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5"
# SAVE_PERIOD = 1
# STEPS_PER_EPOCH = 25000 / 25
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    verbose = 1,
  save_freq= 'epoch')
callbacks_list = [cp_callback]
model.fit(training_generator,epochs=20, validation_data=validation_generator, verbose = 1, callbacks=callbacks_list)
# Save the trained model
model.save('C:/Users/RippS/FrogNetwork')





