import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from getdata import get_csvdata_ADNI, get_csvdata_OASIS, get_slices_ADNI, get_slices_OASIS
from image_processing import segment


def get_tts(N=0, d=1, dim=2, m=None, normalize=True, channels=3, drop=False, segmented=False, multiclass = False):
    if m is None:
        mdict = {0: 90, 1: 110, 2: 90}
        m = mdict[dim]
    df_a = get_csvdata_ADNI(drop_MCI= drop, multiclass=multiclass)
    df_o= get_csvdata_OASIS(drop_young= drop, drop_contradictions=drop, multiclass=multiclass)

    if multiclass:
      df_a_train, df_a_test, y_a_train, y_a_test = train_test_split(df_a["ID"], df_a[['CN', 'MCI', 'AD']], stratify=df_a['label'], random_state=42)
      df_o_train, df_o_test, y_o_train, y_o_test = train_test_split(df_o["ID"], df_o[['CN', 'MCI', 'AD']], stratify=df_o['label'], random_state=42)
    else:
      df_a_train, df_a_test, y_a_train, y_a_test = train_test_split(df_a["ID"], df_a['label'], stratify=df_a['label'], random_state=42)
      df_o_train, df_o_test, y_o_train, y_o_test = train_test_split(df_o["ID"], df_o['label'], stratify=df_o['label'], random_state=42)

    y_o_train = y_o_train.repeat(1+2*N)
    y_a_train = y_a_train.repeat(1+2*N)

    print("loading train OASIS 2D-Data")
    X_train_o = get_slices_OASIS(df_o_train, dim=dim, m=m, N=N, d=d, normalize=normalize)
    print("loading train ADNI 2D-Data")
    X_train_a = get_slices_ADNI(df_a_train, dim=dim, m=m, N=N, d=d, normalize=normalize)

    print("loading test OASIS 2D-Data")
    X_test_o = get_slices_OASIS(df_o_test, dim=dim, m=m, normalize=normalize)
    print("loading test ADNI 2D-Data")
    X_test_a = get_slices_ADNI(df_a_test, dim=dim, m=m, normalize=normalize)

    X_train = np.concatenate((X_train_o, X_train_a), axis=0)
    X_test = np.concatenate((X_test_o, X_test_a), axis=0)

    y_train = np.concatenate((y_o_train, y_a_train))
    y_test = np.concatenate((y_o_test, y_a_test))

    if segmented: 
      print("segmenting data...")
      X_train = np.repeat(X_train[..., np.newaxis], 1, -1)
      X_test = np.repeat(X_test[..., np.newaxis], 1, -1)
      X_train = np.array([segment(x)[:,:,0] for x in tqdm(X_train)])
      X_test = np.array([segment(x)[:,:,0] for x in tqdm(X_test)])
    else:
      X_train = np.repeat(X_train[..., np.newaxis], channels, -1)
      X_test = np.repeat(X_test[..., np.newaxis], channels, -1)

    print("finished loading data")
    dftest = pd.concat([df_o_test, df_a_test], axis=0)
    
    return X_train, X_test, y_train, y_test, dftest


d=2
N=0
X_train0, X_test0, y_train, y_test, dftest = get_tts(dim=0, N=N, d=d, normalize=True, segmented=True, multiclass=False)
X_train1, X_test1, y_train, y_test, dftest = get_tts(dim=1, N=N, d=d, normalize=True, segmented=True, multiclass=False)
X_train2, X_test2, y_train, y_test, dftest = get_tts(dim=2, N=N, d=d, normalize=True, segmented=True, multiclass=False)

print('saving data')
dftest.to_csv("./modelling/predictions.csv", index=False)
np.savez_compressed('./preprocessing/processed_data/segmented_slices.npz',
 X_train0=X_train0, X_test0=X_test0, X_train1=X_train1, X_test1=X_test1,
  X_train2=X_train2, X_test2=X_test2, y_train=y_train, y_test=y_test)