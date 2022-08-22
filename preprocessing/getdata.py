import numpy as np
import pandas as pd
import nibabel as nib
import os

def get_csvdata(drop_young=True, drop_contradictions=True):
    '''
    Loads the .csv dataset and returns a preprocessed dataframe.
        
        Parametes: drop_young (if true, removes entries with age < 60)
        
        Processing steps:
            NaNs in "CDR" are replaced by 0
            Remove entries of young patients (Optional)
            Remove entries where CDR and MMSE results contradict each other
        
        Returns: the processed Dataframe
    '''
    df = pd.read_csv('../data/oasis_cross-sectional.csv')
    df['CDR'].fillna(0, inplace=True)
    if drop_young:
        df=df[df['Age']>=60]
    if drop_contradictions:
        df = df[((df['CDR']==1.0) & (df['MMSE']<29)) | ((df['CDR']==0.5) & (df['MMSE']<30)) | ((df['CDR']==0.0) & (df['MMSE']>26))]
    df['CDR']=(df['CDR']>0).astype(int)
    return df

def get_slices(IDs, N=0, d=1, dim=0, m=95, normalize=True, file="masked"):
    '''
    Returns slices of masked 3D-images at given Paths
        Parameters:
                IDs: list of paths 
                N: number of steps in each direction
                d: step size
                dim: axis along which the image is sliced
                    0= sagittal, 1= cortical, 2= traverse
                m: starting slice

        Returns: 
                len(IDs)*(1+2N) slices 
    '''
    imgs = []
    for path in IDs:
        if file=="segmented":
            path1 = '../data/Oasis_Data/' + path + '/FSL_SEG/'
            for path2 in os.listdir(path1):
                if path2.endswith('fseg.img'):
                    img = nib.load(path1+path2)
        elif file=="masked":
            path1 = '../data/Oasis_Data/' + path + '/PROCESSED/MPRAGE/T88_111/'
            for path2 in os.listdir(path1):
                if path2.endswith('masked_gfc.img'):
                    img = nib.load(path1+path2)
        img = img.get_fdata().take(0,axis=3)
        if normalize:
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
        imgs.append(img)
    return np.array(imgs)