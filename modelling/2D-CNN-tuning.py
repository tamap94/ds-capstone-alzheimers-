import os
from pickle import TRUE
import sys
from datetime import datetime

sys.path.append('../')
print(os.path.abspath("."))
os.chdir(os.path.abspath(".")+"/modelling")
from preprocessing.getdata import *

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from scipy import stats

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, BatchNormalization, RandomCrop, RandomRotation, RandomTranslation, LocallyConnected2D

import logging
from logging import getLogger
logger = logging.getLogger()
logging.getLogger("pyhive").setLevel(logging.CRITICAL) 
logger.setLevel(logging.INFO)

import mlflow 
from config import TRACKING_URI, EXPERIMENT_NAME

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

RSEED=42
np.random.seed(42)
tf.random.set_seed(42)

def get_data(dataset, N=0, Ntest=0, d=1, m=90, dim=2, norm=True, file="masked", drop_young=True, drop_contradictions=True, drop_MCI = True):

    # load csv_file
    if dataset == "OASIS":
        logger.info(f"Loading OASIS-csv, drop_young={drop_young}, drop_contradictions={drop_contradictions}")
        df = get_csvdata(drop_young=drop_young, drop_contradictions=drop_contradictions)
    elif dataset == "ADNI":
        logger.info(f"Loading ADNI-csv, drop_MCI= {drop_MCI}")
        df = get_csvdata_ADNI(drop_MCI = drop_MCI)
    elif dataset == "both":
        logger.info(f"Loading both-csvs, drop_young={drop_young}, drop_contradictions={drop_contradictions}, drop_MCI= {drop_MCI}")
        df_o = get_csvdata(drop_young=drop_young, drop_contradictions=drop_contradictions)
        df_a = get_csvdata_ADNI(drop_MCI = drop_MCI)


    if dataset in ["OASIS", "ADNI"]:
        #split dataframe into train and test
        logger.info("Train test split on dataframe")
        dfTrain, dfTest, y_train, y_test = train_test_split(df, df['label'], stratify = df['label'], random_state=RSEED)
    elif dataset == "both":
        logger.info("Train test split on both dataframes")
        dfTrain_o, dfTest_o, y_train_o, y_test_o = train_test_split(df_o, df_o['label'], stratify = df_o['label'], random_state=RSEED)
        dfTrain_a, dfTest_a, y_train_a, y_test_a = train_test_split(df_a, df_a['label'], stratify = df_a['label'], random_state=RSEED)
        logger.info(f" dfTrain_o.shape:: {dfTrain_o.shape}  dfTest_o.shape: {dfTest_o.shape}  y_train_o.shape: {y_train_o.shape}  y_test_o.shape: {y_test_o.shape}")
        logger.info(f" dfTrain_a.shape: {dfTrain_a.shape}  dfTest_a.shape: {dfTest_a.shape}  y_train_a.shape: {y_train_a.shape}  y_test_a.shape: {y_test_a.shape}")

    mlflow.set_tag("Dataset", dataset)
    logger.info("empty training_history instantiated")
    
    # load image data (N slices above and below the plane (m), suggested by the datasource)
    # standard values for m are 95 for dim=0, 110 for dim=1, 90 for dim=2
    
    if dataset == "OASIS":
        logger.info(f"Loading 2D-OASIS train data: N,d,m,dim,norm,file={N},{d},{m},{dim},{norm},{file}")
        X_train = get_slices(dfTrain['ID'], N=N, d=d, m=m, dim=dim, normalize=norm, file=file)
        
        logger.info(f"Loading 2D-OASIS test data: Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm},{file}")
        X_test = get_slices(dfTest['ID'], m=m, d=d, dim=dim, N=Ntest, normalize=norm, file=file)

        y_train = y_train.repeat(1+2*N)

    elif dataset == "ADNI":
        logger.info(f"Loading 2D-ADNI train data: N,d,m,dim,norm,file={N},{d},{m},{dim},{norm},{file}")
        X_train = get_slices_ADNI(dfTrain['ID'], N=N, d=d, m=m, dim=dim, normalize=norm)
        
        logger.info(f"Loading 2D-ADNI test data: Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm},{file}")
        X_test = get_slices_ADNI(dfTest['ID'], m=m, d=d, dim=dim, N=Ntest, normalize=norm)
        
        y_train = y_train.repeat(1+2*N)
    
    elif dataset == "both":
        logger.info(f"Loading and concatenating both train datasets: N,d,m,dim,norm,file={N},{d},{m},{dim},{norm},{file}")
        X_train = get_slices_both(dfTrain_o['ID'], dfTrain_a["ID"], N=N, d=d, m=m, dim=dim, normalize=norm)
        
        logger.info(f"Loading and concatenating both test datasets: Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm},{file}")
        X_test = get_slices_both(dfTest_o['ID'], dfTest_a["ID"], N=N, d=d, m=m, dim=dim, normalize=norm)
        
        logger.info(f"X_train.shape {X_train.shape}, X_test.shape {X_test.shape}")
        y_train_o = y_train_o.repeat(1+2*N)
        y_train_a = y_train_a.repeat(1+2*N)
        dfTrain= pd.concat([dfTrain_o, dfTrain_a])
        dfTest= pd.concat([dfTest_o, dfTest_a])
        y_train = pd.concat([y_train_o, y_train_a])
        y_test = pd.concat([y_test_o, y_test_a])
        logger.info(f"y_train.shape: {y_train.shape}  y_test.shape {y_test.shape}")

        # shuffle, since ADNI is only concatednated to the end of OASIS
        X_train, y_train  = shuffle(X_train, y_train)
        X_test, y_test, dfTest  = shuffle(X_test, y_test, dfTest)
    
    data_params = f"N,d,m,dim,Ntest,norm,file={N},{d},{m},{dim},{Ntest},{norm},{file}"
    mlflow.log_params({"loading-params": data_params})
    
    '''X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    X_test = np.repeat(X_test[..., np.newaxis], 3, -1) '''
    
    return X_train, X_test, y_train, y_test, dfTest
  
def build_model(X_train, X_test, model_name="..."): #<=========== change this when changing the model architecture
    #with tf.device('/cpu:0'):
    HEIGHT = X_train.shape[1]
    WIDTH = X_train.shape[2]
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    mlflow.tensorflow.autolog()
    logger.info(f"CNN model instantiated: {model_name}")
    mlflow.set_tag("Model Name",model_name)
    
    '''    INPUT_SHAPE = (HEIGHT, WIDTH, 3)
    b_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(b_model)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    # Defining optimizer and learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.00001,
        decay_steps=10000,
        decay_rate=1,
        staircase=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, name='Adam')
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])'''

    model = Sequential()
    # layers
    model.add(InputLayer(input_shape=[HEIGHT, WIDTH, 1], name='image'))
    #model.add(RandomRotation(factor=0.2, fill_mode="reflect", interpolation="bilinear",  seed=None, fill_value=0.0))
    #model.add(LocallyConnected2D(1, 3, strides=3))
    model.add(Conv2D(8, 3, activation="relu", padding="same", kernel_regularizer='l2'))
    model.add(Conv2D(16, 3, activation="relu", padding="same", kernel_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer='l2'))
    model.add(Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu", kernel_regularizer='l2'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, name='Adam')

    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])
    print(model.summary)
        
    return model
    
def fit_and_predict_model(model, X_train, y_train, X_test, Ntest=0, BATCH_SIZE= 32, VAL_SPLIT= 0.2, EPOCHS=25):
    #with tf.device('/cpu:0'):
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
    return y_pred, y_pred_probs

#######################################################################################


today= datetime.now().strftime("%y-%m-%d")
os.makedirs("./predictions/"+today, exist_ok=True)
os.makedirs("./logs/"+today, exist_ok=True)
os.makedirs("./saved_models/"+today, exist_ok=True)
now= datetime.now().strftime("%H:%M:%S")


#######################################################################################

run_name = "few_slices_for_EA" #<============ TODO Change for every run
ds= "both"
normalize=False
filetype= "masked"
drop_y, drop_cont, drop_MCI = False, False, False 
N=0
slices = [(2, 88), (2, 90), (2, 92), (1,90), (1,95), (1,100)]
epochs=30
batch_size=32

########################################################################################
logging.basicConfig(format="%(asctime)s: %(message)s", filename="./logs/"+today+"/"+now+"-"+run_name+".log")

for dim_slice in slices:
    mlflow.start_run(run_name=run_name)
    now_2 = datetime.now().strftime("%H:%M")
    logger.info("Loading data")
    X_train, X_test, y_train, y_test, dfTest = get_data(
        dataset=ds, N=N, Ntest=0, d=1, m=dim_slice[1], dim=dim_slice[0], norm=normalize,
        file=filetype, drop_young=drop_y, drop_contradictions=drop_cont, drop_MCI = drop_MCI) 
    logger.info("Building the model")
    model= build_model(X_train, X_test)
    logger.info("Training and predicting")
    y_pred, y_pred_probs= fit_and_predict_model(model, X_train, y_train, X_test, Ntest=0, BATCH_SIZE= batch_size, VAL_SPLIT= 0.2, EPOCHS=epochs)
    print(dfTest.shape, y_test.shape, y_pred.shape, y_pred_probs.shape)
    dfTest["y_test"], dfTest["y_pred"], dfTest["y_pred_probs"] = y_test, y_pred, y_pred_probs
    dfTest.to_csv("predictions/"+today+"/"+run_name+"-"+now_2+".csv")
    model.save("saved_models/"+today+"/"+run_name+"-"+now_2)
    print(os.path.abspath("."))
    mlflow.end_run()