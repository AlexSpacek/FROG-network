#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
# Ntraces=1
# trace = np.empty([Ntraces,256,256])
# #SizeAndfeatures = np.empty([Ntraces,5])
# #features = np.empty([Ntraces,3])
# labels = np.empty([Ntraces,256])
# for n in range(1,Ntraces+1):
#     TrainingTracePath='C:/Users/alexandr.spacek/FROGnetwork/denseNet/data'+str(n)+'Trace.txt'
#     TrainingLabelsPath='C:/Users/alexandr.spacek/FROGnetwork/denseNet/data'+str(n)+'Label.txt'
#     trace[n-1,:,:] = np.loadtxt(TrainingTracePath,skiprows=1)
#     labels[n-1,] = np.loadtxt(TrainingLabelsPath)
# np.argwhere(np.isnan(trace))
# traces_noNAN = np.delete(trace,57,axis=0) #misto 57 dat indexy NaN
# labels_noNAN = np.delete(labels,57,axis=0)
# features_noNAN = np.delete(features,57,axis=0)
# traces_noNAN = trace #misto 57 dat indexy NaN
# labels_noNAN = labels
# features_noNAN = features
# np.argwhere(np.isnan(traces_noNAN))
# max1 = np.max(abs(labels[0,0]))
# max2 = np.max(abs(labels[0,1]))
# max3 = np.max(abs(labels[0,2]))   
# max4 = np.max(abs(labels[0,3]))          

#labels[0,0] = (labels[0,0])/2e6
#labels[0,1] = ((labels[0,1]/1e9)+1)/2
#labels[0,2] = ((labels[0,2]/1e11)+1)/2
#labels[0,3] = ((labels[0,3]/4))    
    
    
# model = tf.keras.models.load_model('C:/Users/alexandr.spacek/FROGnetwork/denseNet/')
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
#cov1 = Conv2D(24, [5,5], strides=2, activation='elu')(image_input)
#cov2 = Conv2D(36, [5,5], strides=2, activation='elu')(cov1)
#cov3 = Conv2D(64, [3,3], activation='elu')(cov2)

# dropout = Dropout(0.5)(x)
# flatten = Flatten()(dropout)
d2 = Dense(512, activation='linear')(x)
#branchA = Dense(128, activation = 'relu')(d1)
#branchB = Dense(128, activation = 'linear')(d1)
#out = concatenate([branchA, branchB])
out = Dense(128, activation='linear')(d2)
# out = Lambda(lambda x: x**2)(d3)
# model = Model(inputs=[image_input], outputs=[out])
model = Model(inputs=model_d.input, outputs=[out])

model.compile(loss='mse', optimizer='Adam',metrics=['mean_squared_error'])

checkpoint_filepath = 'CheckpointsMorePhaseNoise/weights.13-0.00.hdf5'
model.load_weights(checkpoint_filepath)
predicted = model.predict(TestTrace)
print(predicted[0,])
print(TestLabel[0,])
# test0=trace[0,:,:].reshape(1,256,256)
# predicted = model.predict([test0])
#predictedGDD = max1*(predicted[0,0])
#predictedTOD = max2*(2*(predicted[0,1])-1)
#predictedFOD = max3*(2*(predicted[0,2])-1)
#predictedL = max4*((predicted[0,3]))
#print([predicted])
#print([labels])
# x = np.arange(1,129)
# complexPred = predicted[0,0:128]+1j *predicted[0,128:]
# complexLabel = labels[0,0:128]+1j *labels[0,128:]
#plt.plot(x,np.transpose(predicted[0,0:127])) 
#plt.show()
#plt.plot(x,np.transpose(labels[whichToTest,0:127])) 
#plt.show()
#plt.plot(x,np.abs(complexPred)**2) 
#plt.show()
#plt.plot(x,np.abs(complexLabel)**2) 
#plt.show()
# print(predicted[0,])
# print(labels[0,])

# In[ ]:




