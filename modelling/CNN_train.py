import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Concatenate, Input
np.random.seed(42)
tf.random.set_seed(42)

import mlflow 
from config_img import TRACKING_URI, EXPERIMENT_NAME
print(TRACKING_URI)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

data = np.load('./preprocessing/processed_data/segmented_slices.npz')

X_train0=data['X_train0']
X_test0=data['X_test0']
X_train1=data['X_train1']
X_test1=data['X_test1']
X_train2=data['X_train2']
X_test2=data['X_test2']
y_train=data['y_train']
y_test=data['y_test']

Input0 = Input(shape=X_train0[0].shape, name='input0')
Input1 = Input(shape=X_train1[0].shape, name='input1')
Input2 = Input(shape=X_train2[0].shape, name='input2')

def build_model(Input):
  b_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  for layer in b_model.layers:
      layer.trainable=False
  b_model = Model(inputs=b_model.input, outputs = b_model.layers[-1].output)
  x = Input
  x = b_model(x)
  x = Flatten()(x)
  x = Dropout(0.3)(x)
  x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
  return x

##########

mlflow.start_run()

x = Concatenate()([build_model(Input0), build_model(Input1), build_model(Input2)])
x = Dropout(0.3)(x)
x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[Input0, Input1, Input2], outputs=out)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=100,
    decay_rate=1,
    staircase=False)
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
   boundaries = [600, 1400], 
   values = [1e-3, 1e-4, 1e-5]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, name='Adam')
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']) #tf.keras.losses.BinaryFocalCrossentropy()

mlflow.tensorflow.autolog()

with tf.device('/device:GPU:0'):
  training = model.fit([X_train0, X_train1, X_train2], y_train, epochs=50, batch_size=32, validation_data=([X_test0, X_test1, X_test2], y_test))
print("Training finished, saving the model under 'models/best_model'")
model.save('models/best_model')
pd.DataFrame(training.history).to_csv('modelling/CNN_history.csv')

#save predictions
print('saving predictions')
dftest = pd.read_csv('./modelling/CNN_predictions.csv')
y_pred = model.predict([X_test0, X_test1, X_test2])
dftest['pred'] = y_pred
dftest['y_test'] = y_test
dftest.to_csv('./modelling/CNN_predictions.csv')

mlflow.end_run()