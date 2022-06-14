import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from io import StringIO   # StringIO behaves like a file object
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from keras.layers import ( Conv2D, Flatten, Lambda, Dense, concatenate,
                         Dropout, Input )
from keras.models import Model
from sklearn.utils import class_weight
from my_classes import DataGenerator
    

params = {'dim': (256,256),
          'batch_size': 1,
          'n_classes': 128,
          'n_channels': 1,
          'shuffle': True}
nTraining = 1
nValidation = 0
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
# validation_generator = DataGenerator(partition['validation'], labels['validation'], **params)
[TestTrace, TestLabel] = training_generator[0]
from tensorflow.keras.applications import DenseNet121
image_input = Input(shape=(256, 256, 3))
model_d = DenseNet121(weights='imagenet', include_top=False, pooling='max', classes=1024, input_shape=(256, 256, 3))
x = model_d.output
d2 = Dense(512, activation='linear')(x)
out = Dense(128, activation='linear')(d2)
model = Model(inputs=model_d.input, outputs=[out])

model.compile(loss='mse', optimizer='Adam',metrics=['mean_squared_error'])

checkpoint_filepath = 'CheckpointsMorePhaseNoise/weights.13-0.00.hdf5'
model.load_weights(checkpoint_filepath)
predicted = model.predict(TestTrace)
print(predicted[0,])
print(TestLabel[0,])