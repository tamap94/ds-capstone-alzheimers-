import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
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

def get_csvdata_ADNI():
    '''
    Loads the .csv dataset and returns a preprocessed dataframe.
        
        Parametes: drop_young (if true, removes entries with age < 60)
        
        Processing steps:
            Sort by Subject ID
            Rename column "Subject" to "ID"
            Remove entries of young patients (Optional)
            Remove entries where CDR and MMSE results contradict each other
        
        Returns: the processed Dataframe
    '''
    df = pd.read_csv("../data/ADNI_Freesurfer/FreeSurfer_8_23_2022.csv").sort_values(["Subject","Description"])
    df.rename(columns={"Subject":"ID"}, inplace=True)
    df= df[(df["Description"] != "FreeSurfer Cross-Sectional Processing aparc+aseg") & (df["Description"] != "FreeSurfer Longitudinal Processing aparc+aseg")]
    image_IDs = []
    for i in df["ID"].unique():
        image_IDs.append(df[df["ID"]==i]["Image Data ID"].iloc[0])
    df= df.loc[df["Image Data ID"].isin(image_IDs)]
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
        for i in range(1,N+1): #rotate to match oasis
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

def get_3D_data_ADNI(IDs):
    imgs = []
    for path in IDs:
        path1 = '../data/ADNI_Freesurfer/ADNI/' + path + "/FreeSurfer_Cross-Sectional_Processing_brainmask/"
        try: 
            path2 = path1+os.listdir(path1)[0]
        except:
            path1 = '../data/ADNI_Freesurfer/ADNI/' + path + "/FreeSurfer_Longitudinal_Processing_brainmask/"
            path2 = path1+os.listdir(path1)[0]
        path3 = path2+"/"+os.listdir(path2)[0]
        for file_path in os.listdir(path3):
            if file_path.endswith('brainmask.mgz'):
                img = nib.load(path3+"/"+file_path)
        img = img.get_fdata()
        img = img[35:211,15:191,10:218]
        imgs.append(img)
        imgs= np.array(imgs)
        imgs = np.rot90(imgs, k=3, axes=(1,2))
        imgs = np.rot90(imgs, k=2, axes=(1,2))
    return np.array(imgs)


def get_slices_ADNI(IDs, N=0, d=1, dim=0, m=95, normalize=True):
    '''
    Returns slices of masked 3D-images at given Paths
        Parameters:
                IDs: list of paths 
                N: number of steps in each direction
                d: step size
                dim: axis along which the image is sliced
                    0= sagittal, 1= cortical, 2= traverse
                m: starting slice
                
                rotates the images by 180 degrees to fit with the oasis data

        Returns: 
                len(IDs)*(1+2N) slices 
    '''
    if dim == 1:  #change meaning of imput dimensions to fit oasis data
        dim = 2
    elif dim == 2:
        dim = 1
        m= 176-m
    elif dim== 0:
        m= 176-m
        
    imgs = []
    for path in IDs:
        path1 = '../data/ADNI_Freesurfer/ADNI/' + path + "/FreeSurfer_Cross-Sectional_Processing_brainmask/"
        try:    #for some subjects, only a Longitudinal image is available
            path2 = path1+os.listdir(path1)[0] 
        except:
            path1 = '../data/ADNI_Freesurfer/ADNI/' + path + "/FreeSurfer_Longitudinal_Processing_brainmask/"
            path2 = path1+os.listdir(path1)[0]
        path3 = path2+"/"+os.listdir(path2)[0]
        for file_path in os.listdir(path3):
            if file_path.endswith('brainmask.mgz'):
                img = nib.load(path3+"/"+file_path)
        img = img.get_fdata()
        img = img[35:211,15:191,10:218]
        if normalize:
            if img.max() > 0.0:
                img = img/img.max()
        imgs.append(img.take(m, axis=dim))
        for i in range(1,N+1):
            imgs.append(img.take(m+d*i, axis=dim))
            imgs.append(img.take(m-d*i, axis=dim))
    imgs = np.array(imgs) #rotate images to align with oasis data
    if dim ==0:
        imgs = np.rot90(imgs, k=3, axes=(1,2))
    elif dim ==2:
        imgs = np.rot90(imgs, k=2, axes=(1,2))
    return imgs
