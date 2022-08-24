import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os

def get_csvdata():
    df = pd.read_csv('../data/oasis_cross-sectional.csv')
    df['CDR'].fillna(0, inplace=True)
    df=df[df['Age']>=60]
    df = df[((df['CDR']==1.0) & (df['MMSE']<29)) | ((df['CDR']==0.5) & (df['MMSE']<30)) | ((df['CDR']==0.0) & (df['MMSE']>26))]
    df['CDR']=(df['CDR']>0).astype(int)
    return df

def get_slices(IDs, N=0, d=2, dim=0, m=100):
    '''
    Returns slices of 3D-images at given Paths
        Parameters:
                IDs: list of paths 
                N: number of steps in each direction
                d: step size
                dim: axis along which the image is sliced
                m: starting slice

        Returns: 
                len(IDs)*(1+2N) slices 
    '''
    imgs = []
    for path in IDs:
        path1 = '../data/Oasis_Data/' + path + '/PROCESSED/MPRAGE/T88_111/'
        for path2 in os.listdir(path1):
            if path2.endswith('masked_gfc.img'):
                img = nib.load(path1+path2)
        img = img.get_fdata().take(0,axis=3)
        if img.max() > 0.0:
            img = img/img.max()
        imgs.append(img.take(m, axis=dim))
        for i in range(1,N+1):
            imgs.append(img.take(m+d*i, axis=dim))
            imgs.append(img.take(m-d*i, axis=dim))
    return np.array(imgs)

def get_3D_data(IDs):
    imgs = []
    for path in IDs:
        path1 = '../data/Oasis_Data/' + path + '/PROCESSED/MPRAGE/T88_111/'
        for path2 in os.listdir(path1):
            if path2.endswith('masked_gfc.img'):
                img = nib.load(path1+path2)
        img = img.get_fdata()
        if img.max() > 0.0:
            img = img/img.max()
        imgs.append(img)
    return np.array(imgs)

def get_kaggle(TYPE='binary'):
    path_train = '../data/Alzheimer_s Dataset/train/'
    path_test = '../data/Alzheimer_s Dataset/test/'

    dem = {'NonDemented': 0, 'VeryMildDemented': 1, 'MildDemented': 2, 'ModerateDemented': 3}

    if TYPE == 'regression':
        dem = {'NonDemented': 0.0, 'VeryMildDemented': 0.25, 'MildDemented': 0.5, 'ModerateDemented': 1.0}
    elif TYPE == 'multiclass':
        for i, d in enumerate(dem):
            c = np.zeros(4, dtype=np.int64)
            c[i] = 1
            dem[d] = c
    elif TYPE == 'binary':
        for d in dem:
            if d == 'NonDemented':
                dem[d] = 0
            else: dem[d] = 1


    def read_images(path):
        X = []
        y = []
        for d in dem:
            for img in os.listdir(path+d):
                X.append(plt.imread(path+d+'/'+img)/255.)
                y.append(dem[d])
        X = np.array(X)
        y = np.array(y)
        return X, y

    X_train, y_train = read_images(path_train)
    X_test, y_test = read_images(path_test)

    return X_train, X_test, y_train, y_test