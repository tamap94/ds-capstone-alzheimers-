import pandas as pd
import numpy as np
import sys, os, logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, BatchNormalization, RandomCrop, RandomRotation, RandomTranslation, LocallyConnected2D

sys.path.append('../')
from data_loading import get_csvdata_ADNI, get_csvdata_OASIS, get_slices_both

RSEED = 42
np.random.seed(42)
tf.random.set_seed(42)

logger = logging.getLogger()
logging.getLogger("pyhive").setLevel(logging.CRITICAL) 
logger.setLevel(logging.INFO)

today= datetime.now().strftime("%y-%m-%d")
now= datetime.now().strftime("%H:%M")


def loading(m_dim):
    #loading data tables:
    logger.info("Loading the data table")
    df_o = get_csvdata_OASIS(drop_young=True, drop_contradictions=True)
    df_a = get_csvdata_ADNI(drop_MCI=False)

    #Train test split:
    logger.info("Train test split")
    dfTrain_o, dfTest_o, y_train_o, y_test_o = train_test_split(df_o, df_o['label'], stratify = df_o['label'], random_state=RSEED)
    dfTrain_a, dfTest_a, y_train_a, y_test_a = train_test_split(df_a, df_a['label'], stratify = df_a['label'], random_state=RSEED)
    logger.info(f"dfTrain_o.shape:: {dfTrain_o.shape}  dfTest_o.shape: {dfTest_o.shape}  y_train_o.shape: {y_train_o.shape}  y_test_o.shape: {y_test_o.shape}")
    logger.info(f"dfTrain_a.shape: {dfTrain_a.shape}  dfTest_a.shape: {dfTest_a.shape}  y_train_a.shape: {y_train_a.shape}  y_test_a.shape: {y_test_a.shape}")


    #Loading 2D slices using the IDs from the train-test split dataframe
    N = 0
    Ntest=0
    d = 1
    norm =  True
    m = m_dim[0]
    dim= m_dim[1]

    logger.info(f"Loading and concatenating corresponding iamge train-datasets with N,d,m,dim,norm,file={N},{d},{m},{dim},{norm}")
    X_train = get_slices_both(
        dfTrain_o['ID'], dfTrain_a["ID"], N=N, d=d, m=m, dim=dim, normalize=norm)
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"Loading and concatenating corresponding image test-datasets with Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm}")
    X_test = get_slices_both(
        dfTest_o['ID'], dfTest_a["ID"], N=0, d=d, m=m, dim=dim, normalize=norm)
    logger.info(f"X_test.shape {X_test.shape}")

    #Repeat entries in y to fit with the loading of multiple slices
    y_train_o = y_train_o.repeat(1+2*N)
    y_train_a = y_train_a.repeat(1+2*N)

    #Concatenating and shuffling dataframes, y and X
    dfTrain= pd.concat([dfTrain_o, dfTrain_a])
    dfTest= pd.concat([dfTest_o, dfTest_a])
    y_train = pd.concat([y_train_o, y_train_a])
    y_test = pd.concat([y_test_o, y_test_a])
    logger.info(f"y_train.shape: {y_train.shape}  y_test.shape: {y_test.shape}")
    X_train, y_train  = shuffle(X_train, y_train, random_state=RSEED)
    X_test, y_test, dfTest  = shuffle(X_test, y_test, dfTest)

    #repeat entries in X to fit with the input shape of the pretrained VGG16 model
    logger.info("Adapting X_train and X_test to fit with input shape of pretrained Model")
    X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    logger.info(f"X_train shape: {X_train.shape}")
    X_test = np.repeat(X_test[..., np.newaxis], 3, -1)
    logger.info(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, dfTest
 
def training(X_train, y_train, m_dim):
    #Building the model    
    logger.info(f"Instantiating the CNN model")
    HEIGHT = X_train.shape[1]
    WIDTH = X_train.shape[2]
    INPUT_SHAPE = (HEIGHT, WIDTH, 3)

    #base model
    b_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    for layer in b_model.layers:
        layer.trainable = False

    #keras sequential CNN
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(b_model)
    model.add(BatchNormalization())
    model.add(LocallyConnected2D(4, 3, 1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation="sigmoid"))

    # Defining optimizer and learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.00001,
        decay_steps=10000,
        decay_rate=1,
        staircase=False)
    callback= tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  
        min_delta=0,
        patience=8,   
        restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, 
        name='Adam')
    model.compile(
        optimizer = optimizer, 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])

    BATCH_SIZE=32
    VAL_SPLIT=0.2
    EPOCHS=10

    logger.info(f"Fitting model: batch_size={BATCH_SIZE},validation_split={VAL_SPLIT},epochs={EPOCHS}")

    model.fit(
        X_train, y_train, 
        batch_size = BATCH_SIZE, 
        validation_split=VAL_SPLIT, 
        epochs = EPOCHS,
        callbacks= callback)

    model.save("../models/"+today+"/"+now+"m_dim-"+str(m_dim[0])+"_"+str(m_dim[1]))
    logger.info("Model was saved as m_dim-"+str(m_dim[0])+"_"+str(m_dim[1]))
    return model

def loading_for_combined_model(m_dim, imputed =True):
    #loading data tables:
    logger.info("Loading the numerical data table")
    df_train = pd.read_csv()
    df_test = 
    

    #Train test split:
    logger.info("Train test split")
    dfTrain_o, dfTest_o, y_train_o, y_test_o = train_test_split(df_o, df_o['label'], stratify = df_o['label'], random_state=RSEED)
    dfTrain_a, dfTest_a, y_train_a, y_test_a = train_test_split(df_a, df_a['label'], stratify = df_a['label'], random_state=RSEED)
    logger.info(f"dfTrain_o.shape:: {dfTrain_o.shape}  dfTest_o.shape: {dfTest_o.shape}  y_train_o.shape: {y_train_o.shape}  y_test_o.shape: {y_test_o.shape}")
    logger.info(f"dfTrain_a.shape: {dfTrain_a.shape}  dfTest_a.shape: {dfTest_a.shape}  y_train_a.shape: {y_train_a.shape}  y_test_a.shape: {y_test_a.shape}")


    #Loading 2D slices using the IDs from the train-test split dataframe
    N = 0
    Ntest=0
    d = 1
    norm =  True
    m = m_dim[0]
    dim= m_dim[1]

    logger.info(f"Loading and concatenating corresponding iamge train-datasets with N,d,m,dim,norm,file={N},{d},{m},{dim},{norm}")
    X_train = get_slices_both(
        dfTrain_o['ID'], dfTrain_a["ID"], N=N, d=d, m=m, dim=dim, normalize=norm)
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"Loading and concatenating corresponding image test-datasets with Ntest,d,m,dim,norm,file={Ntest},{d},{m},{dim},{norm}")
    X_test = get_slices_both(
        dfTest_o['ID'], dfTest_a["ID"], N=0, d=d, m=m, dim=dim, normalize=norm)
    logger.info(f"X_test.shape {X_test.shape}")

    #Repeat entries in y to fit with the loading of multiple slices
    y_train_o = y_train_o.repeat(1+2*N)
    y_train_a = y_train_a.repeat(1+2*N)

    #Concatenating and shuffling dataframes, y and X
    dfTrain= pd.concat([dfTrain_o, dfTrain_a])
    dfTest= pd.concat([dfTest_o, dfTest_a])
    y_train = pd.concat([y_train_o, y_train_a])
    y_test = pd.concat([y_test_o, y_test_a])
    logger.info(f"y_train.shape: {y_train.shape}  y_test.shape: {y_test.shape}")
    X_train, y_train  = shuffle(X_train, y_train, random_state=RSEED)
    X_test, y_test, dfTest  = shuffle(X_test, y_test, dfTest)

    #repeat entries in X to fit with the input shape of the pretrained VGG16 model
    logger.info("Adapting X_train and X_test to fit with input shape of pretrained Model")
    X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    logger.info(f"X_train shape: {X_train.shape}")
    X_test = np.repeat(X_test[..., np.newaxis], 3, -1)
    logger.info(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, dfTest




    


