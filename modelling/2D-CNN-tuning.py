import numpy as np
import pandas as pd
import os
import sys
import pickle
sys.path.append('../')
print(os.path.abspath("."))
os.chdir(os.path.abspath(".")+"/modelling")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D
np.random.seed(42)
tf.random.set_seed(42)
from scipy import stats

import logging
from logging import getLogger
import mlflow


logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)


from preprocessing.getdata import *
from config import TRACKING_URI, EXPERIMENT_NAME
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def get_data(dataset, N=0, Ntest=0, d=1, m=90, dim=2, norm=True, file="masked", drop_young=True, drop_contradictions=True, drop_MCI = True):

    # load csv_file
    if dataset == "OASIS":
        logger.info(f"Loading OASIS-csv, drop_young=True, drop_contradictions=True")
        df = get_csvdata(drop_young=drop_young, drop_contradictions=drop_contradictions)
    elif dataset == "ADNI":
        logger.info(f"Loading ADNI-csv, drop_MCI= True")
        df = get_csvdata_ADNI(drop_MCI = drop_MCI)
    else:
        logger.error("no valid dataset speficied")

    #split dataframe into train and test
    logger.info("Train test split on dataframe")
    dfTrain, dfTest, y_train, y_test = train_test_split(df, df['label'], stratify = df['label'], random_state=42)

    mlflow.set_tag("Dataset", dataset)
    logger.info("empty training_history instantiated")
    
    # load image data (N slices above and below the plane (m), suggested by the datasource)
    # standard values for m are 95 for dim=0, 110 for dim=1, 90 for dim=2

    if dataset == "OASIS":
        logger.info(f"Loading 2D-OASIS train data: N,d,m,dim,norm,file={N},{d},{m},{dim},{norm},{file}")
        X_train = get_slices(dfTrain['ID'], N=N, d=d, m=m, dim=dim, normalize=norm, file=file)
        
        logger.info(f"Loading 2D-OASIS test data: Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm},{file}")
        X_test = get_slices(dfTest['ID'], m=m, d=d, dim=dim, N=Ntest, normalize=norm, file=file)

    elif dataset == "ADNI":
        logger.info(f"Loading 2D-OASIS train data: N,d,m,dim,norm,file={N},{d},{m},{dim},{norm},{file}")
        X_train = get_slices_ADNI(dfTrain['ID'], N=N, d=d, m=m, dim=dim, normalize=norm)
        
        logger.info(f"Loading 2D-OASIS test data: Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm},{file}")
        X_test = get_slices_ADNI(dfTest['ID'], m=m, d=d, dim=dim, N=Ntest, normalize=norm)
    
    y_train = y_train.repeat(1+2*N) 
    data_params = f"N,d,m,dim,Ntest,norm,file={N},{d},{m},{dim},{Ntest},{norm},{file}"
    mlflow.log_params({"loading-params": data_params})
    return X_train, X_test, y_train, y_test
  
def build_model(X_train, model_name="CNN_8-16_lessReg"):
    HEIGHT = X_train.shape[1]
    WIDTH = X_train.shape[2]
    mlflow.tensorflow.autolog()

    model = Sequential()
    logger.info(f"CNN model instantiated: {model_name}")
    mlflow.set_tag("Model Name",model_name)
    # layers
    model.add(InputLayer(input_shape=[HEIGHT, WIDTH, 1], name='image'))
    model.add(Conv2D(8, 3, activation="relu", padding="same"))
    model.add(Conv2D(8, 3, activation="relu", padding="same"))
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Conv2D(16, 3, activation="relu", padding="same"))
    model.add(Conv2D(16, 3, activation="relu", padding="same"))
    #model.add(Dropout(0.2))
    #model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    #model.add(Conv2D(32, 3, activation="relu", padding="same"))
    #model.add(Conv2D(32, 3, activation="relu", padding="same"))
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu", kernel_regularizer='l2'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, name='Adam')

    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])#, keras.metrics.Precision(), keras.metrics.Recall()])
    print(model.summary())

    return model
    
def fit_and_predict_model(model, X_train, y_train, X_test, Ntest=0, BATCH_SIZE= 32, VAL_SPLIT= 0.2, EPOCHS=25):
    logger.info(f"Fitting model and storing history: batch_size={BATCH_SIZE},validation_split={VAL_SPLIT},epochs={EPOCHS}")
    model.fit(X_train, y_train, batch_size = BATCH_SIZE, validation_split=VAL_SPLIT, epochs = EPOCHS)
    
    # prediction of outcomes and conversion to binary
    logger.info("Predicting on Xtest")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs>0.5).astype(int)
    # reshape is necessary if X_test also consists of multiple slices per sample

    if Ntest != 0:
        test_length, slices_per_test_sample = dfTest.shape[0], int(y_pred.shape[0]/dfTest.shape[0])

        l = []
        for i in range(test_length):
            for j in np.where(y_pred.reshape((test_length,slices_per_test_sample))[i]==y_test.values[i])[0]:
                l.append(j)

        y_pred = np.array(stats.mode(y_pred.reshape((test_length,slices_per_test_sample)), axis=1, keepdims=False))[0]
        
    logger.info("Measuring prediction performance and storing on ML-Flow")
    mlflow.log_metric("test" + "-" + "acc", accuracy_score(y_test, y_pred.round()).round(2))
    mlflow.log_metric("test" + "-" + "recall", recall_score(y_test, y_pred.round()).round(2))
    mlflow.log_metric("test" + "-" + "precision", precision_score(y_test, y_pred.round()).round(2))

#-----------------------------------------------------------------------

for slice in range(80:90):
    mlflow.start_run()
    X_train, X_test, y_train, y_test = get_data(
        dataset="ADNI",N=0, Ntest=0, d=1, m=slice, dim=2, norm=True,
         file="masked", drop_young=True, drop_contradictions=True, drop_MCI = True) 
    
    model= build_model(X_train, model_name="CNN_8-16_lessReg")
    fit_and_predict_model(model, X_train, y_train, X_test, Ntest=0, BATCH_SIZE= 32, VAL_SPLIT= 0.2, EPOCHS=25)
    mlflow.end_run()