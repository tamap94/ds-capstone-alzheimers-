import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os
from logging import getLogger
from tqdm import tqdm
from dipy.segment.tissue import TissueClassifierHMRF

from sklearn.model_selection import train_test_split

def get_csvdata(drop_young=True, drop_contradictions=True, multiclass = False):
    '''
    Loads the .csv dataset and returns a preprocessed dataframe.
        
        Parameters: drop_young (if true, removes entries with age < 60)
        
        Processing steps:
            NaNs in "CDR" are replaced by 0
            Remove entries of young patients (Optional)
            Remove entries where CDR and MMSE results contradict each other
            Drops the "Delay" and "Hand" columns 
        
        Returns: the processed Dataframe
    '''
    df = pd.read_csv('../data/oasis_cross-sectional.csv')
    df['CDR'].fillna(0, inplace=True)
    if drop_young:
        df=df[df['Age']>=30]
    if drop_contradictions:
        df = df[((df['CDR']==1.0) & (df['MMSE']<29)) | ((df['CDR']==0.5) & (df['MMSE']<30)) | ((df['CDR']==0.0) & (df['MMSE']>26))]
    df.drop(labels=['Delay', 'Hand'], axis=1, inplace=True)
    df['label']=(df['CDR']>0).astype(int)
    if multiclass:
        df = df.join(pd.get_dummies(df["CDR"].replace({0.0:"CN", 0.5:"MCI", 1.0:"AD", 2.0:"AD"})))
        def label(row):
            if row.CN == 1:
                return "CN"
            if row.MCI == 1:
                return "MCI"
            if row.AD == 1:
                return "AD"
        df["label"] = df.apply(lambda row: label(row), axis=1)
    df["dataset"] = "OASIS"
    return df

def get_csvdata_ADNI(drop_MCI = True, multiclass = False):
    '''
    Loads the .csv dataset and returns a preprocessed dataframe.
        
        Parametes: drop_young (if true, removes entries with age < 60)
        
        Processing steps:
            Sort by Subject ID
            Rename column "Subject" to "ID"
            adds a column "label" 
        
        Returns: the processed Dataframe
    '''
    df = pd.read_csv("../data/ADNI_Freesurfer/FreeSurfer_8_23_2022.csv").sort_values(["Subject","Description"])
    df.rename(columns={"Subject":"ID"}, inplace=True)
    df= df[(df["Description"] != "FreeSurfer Cross-Sectional Processing aparc+aseg") & (df["Description"] != "FreeSurfer Longitudinal Processing aparc+aseg")]
    image_IDs = []
    for i in df["ID"].unique():
        image_IDs.append(df[df["ID"]==i]["Image Data ID"].iloc[0])
    df= df.loc[df["Image Data ID"].isin(image_IDs)]
    #logger.info("ADNI-csv loaded")
    if drop_MCI:
        df= df[(df["Group"] == "AD") | (df["Group"] == "CN")]
        df["label"] = df["Group"] == "AD"
    df["label"] = ((df["Group"] == "AD") | (df["Group"] == "MCI")).astype(int)
    if multiclass:
        df = df.join(pd.get_dummies(df["Group"]))
        def label(row):
            if row.CN == 1:
                return "CN"
            if row.MCI == 1:
                return "MCI"
            if row.AD == 1:
                return "AD"
        df["label"] = df.apply(lambda row: label(row), axis=1)
    df["dataset"] = "ADNI"
    return df

def rename_ADNI(IDs):
    '''renames all 3D brainsmask files to also contain the SubjectID'''
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
                os.rename(path3+'/brainmask.mgz', path3+"/"+path+"-brainmask.mgz")

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
    for path in tqdm(IDs):
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
        img = np.asarray(img.dataobj).take(0,axis=3)
        imgs.append(img.take(m, axis=dim))
        for i in range(1,N+1): #rotate to match oasis
            imgs.append(img.take(m+d*i, axis=dim))
            imgs.append(img.take(m-d*i, axis=dim))
    imgs = np.array(imgs)
    if normalize:
        imgs = imgs/imgs.max()
    #logger.info("OASIS 2D-Data loaded")
    return imgs



def get_3D_data(IDs, normalize=True):
    imgs = []
    for path in tqdm(IDs):
        path1 = '../data/Oasis_Data/' + path + '/PROCESSED/MPRAGE/T88_111/'
        for path2 in os.listdir(path1):
            if path2.endswith('masked_gfc.img'):
                img = nib.load(path1+path2)
        img = np.asarray(img.dataobj).take(0,axis=3)
        imgs.append(img)
    imgs = np.array(imgs)
    if normalize:
        imgs = imgs/imgs.max()
    #logger.info("OASIS 3D-Data loaded")
    return imgs

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

def get_ADNI_dataobj(path):
    path1 = '../data/ADNI_Freesurfer/ADNI/' + path + "/FreeSurfer_Cross-Sectional_Processing_brainmask/"
    foundimg = False
    for root, dirs, files in os.walk(path1):
        for filee in files: 
            if filee.endswith('brainmask.mgz'):
                img = nib.load(root+"/"+filee)
                foundimg = True
    if foundimg == False:
        path1 = '../data/ADNI_Freesurfer/ADNI/' + path + "/FreeSurfer_Longitudinal_Processing_brainmask/"
        for root, dirs, files in os.walk(path1):
            for filee in files: 
                if filee.endswith('brainmask.mgz'):
                    img = nib.load(root+"/"+filee)
                    foundimg = True
    if foundimg == False:
        print('could not find: ', path)
        return False
    else: 
        return img

def get_3D_data_ADNI(IDs, normalize=True):
    imgs = []
    for path in tqdm(IDs):
        img = get_ADNI_dataobj(path)
        if img == False:
            continue
        img = np.asarray(img.dataobj[35:211,15:191,10:218])
        img = np.rot90(img, k=2, axes=(0,1))
        img = np.rot90(img, k=3, axes=(1,2))
        img = np.rot90(img, k=2, axes=(0,2))
        imgs.append(img)
    imgs= np.array(imgs)
    if normalize:
        imgs = imgs/imgs.max()
    #logger.info("ADNI 3D-Data loaded")
    return imgs

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
    imgs = []
    for path in tqdm(IDs):
        img = get_ADNI_dataobj(path)
        if img == False:
            continue
        img = np.asarray(img.dataobj[35:211,15:191,10:218])
        img = np.rot90(img, k=2, axes=(0,1))
        img = np.rot90(img, k=3, axes=(1,2))
        img = np.rot90(img, k=2, axes=(0,2))
        imgs.append(img.take(m, axis=dim))
        for i in range(1,N+1):
            imgs.append(img.take(m+d*i, axis=dim))
            imgs.append(img.take(m-d*i, axis=dim))
    imgs = np.array(imgs) 
    if normalize:
        imgs = imgs/imgs.max()
    return imgs

def get_slices_both(OASIS_IDs, ADNI_IDs, N=0, d=1, dim=0, m=95, normalize=True,  file="masked"):
    imgs_OASIS = get_slices(IDs= OASIS_IDs, N=N, d=d, dim=dim, m=m, normalize=normalize, file=file)
    imgs_ADNI = get_slices_ADNI_new(IDs= ADNI_IDs, N=N, d=d, dim=dim, m=m, normalize=normalize)
    return np.concatenate((imgs_OASIS, imgs_ADNI))


def get_tadpole(drop_MCI = True):
    '''
    Loads the .csv dataset and returns a preprocessed dataframe.
        
        Parametes: drop_young (if true, removes entries with age < 60)
        
        Processing steps:
            Sort by Subject ID
            Rename column "Subject" to "ID"
            adds a column "label" 
            only takes the entries at the first visit 
            drop all the columns without information 
        
        Returns: the processed Dataframe
    '''
    df = pd.read_csv("../tadpole_challenge/ADNIMERGE.csv")
    df.rename(columns={"PTID":"ID"}, inplace=True)
    df= df[(df['Month']==0) & (df['COLPROT'] == "ADNI1")]
    
    #logger.info("ADNI-csv loaded")
    if drop_MCI:
        df= df[(df["DX_bl"] == "AD") | (df["DX_bl"] == "CN")]
        df["label"] = df["DX_bl"] == "AD"
    df["label"] = (df["DX_bl"] == "AD") | (df["DX_bl"] == "MCI")
    df["label"]=df["label"].astype(int)
    return df


def drop_tadpole(df): 
    '''
    Drops all the columns that are not used for the EDA or the modeling 
    '''
    col = ['FDG','AV45', 'CDRSB',  'MidTemp','DX','RID', 'VISCODE', 'SITE', 'COLPROT', 'ORIGPROT', 'EXAMDATE', 'DX_bl', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'PIB', 'ADASQ4', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'FLDSTRENG', 'FSVERSION', 'IMAGEUID',  'Fusiform',  'ICV', 'mPACCdigit', 'mPACCtrailsB', 'EXAMDATE_bl', 'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'LDELTOTAL_BL', 'DIGITSCOR_bl', 'TRABSCOR_bl', 'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl', 'FLDSTRENG_bl', 'FSVERSION_bl', 'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl', 'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'FDG_bl', 'PIB_bl', 'AV45_bl', 'Years_bl', 'Month_bl', 'Month', 'M', 'update_stamp']
    df.drop(columns=col, inplace=True, axis=1)
    return df 

def col_tadpole(df): 
    df["PTAU"].replace("<8",np.nan, inplace=True)
    df["PTAU"].replace(">120",np.nan, inplace=True)
    df["ABETA"].replace("<200",np.nan, inplace=True)
    df["ABETA"].replace(">1700",np.nan, inplace=True)
    df["TAU"].replace("<80",np.nan, inplace=True)
    df["TAU"].replace(">1300",np.nan, inplace=True)
    return df 


def get_tts(N=0, d=1, dim=2, m=None, normalize=True, channels=3, drop=False, segmented=False):
    if m is None:
        mdict = {0: 90, 1: 110, 2: 90}
        m = mdict[dim]
    df_a = get_csvdata_ADNI(drop)
    df_o= get_csvdata(drop, drop)

    df_a_train, df_a_test, y_a_train, y_a_test = train_test_split(df_a['ID'], df_a['label'], stratify=df_a['label'], random_state=42)
    df_o_train, df_o_test, y_o_train, y_o_test = train_test_split(df_o['ID'], df_o['label'], stratify=df_o['label'], random_state=42)

    y_o_train = y_o_train.repeat(1+2*N)
    y_a_train = y_a_train.repeat(1+2*N)

    print("loading train OASIS 2D-Data")
    X_train_o = get_slices(df_o_train, dim=dim, m=m, N=N, d=d, normalize=normalize)
    print("loading train ADNI 2D-Data")
    X_train_a = get_slices_ADNI_new(df_a_train, dim=dim, m=m, N=N, d=d, normalize=normalize)

    print("loading test OASIS 2D-Data")
    X_test_o = get_slices(df_o_test, dim=dim, m=m, normalize=normalize)
    print("loading test ADNI 2D-Data")
    X_test_a = get_slices_ADNI_new(df_a_test, dim=dim, m=m, normalize=normalize)

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
    return X_train, X_test, y_train, y_test

def crop_adni3D_trav(img):
    '''crops an ADNI 3D image to fit the brain center in slice 165 with the center of the OASIS images'''
    cy_o, cx_o, = 86,103
    (X, Y) = img.shape[1:3]
    m = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            m[x, y] = img[(165, x, y)] != 0
    m = m / np.sum(np.sum(m))

    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    cx_a = np.sum(dx * np.arange(X)).astype(int)
    cy_a = np.sum(dy * np.arange(Y)).astype(int)
    if cx_a < 86:
        cx_a = 86
    if cy_a < 103:
        cy_a = 103
    img_crop = img[:, (cx_a-cy_o) : (cx_a-cy_o+176), (cy_a-cx_o) : (cy_a-cx_o+208)]

    return img_crop

def crop_adni3D_cort(img):
    '''crops an ADNI 3D image to fit the brain center in slice 130 with the center of the OASIS images'''
    cy_o = 86
    (X, Y) = img.shape[1:3]
    m = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            m[x, y] = img[(130, x, y)] != 0
    m = m / np.sum(np.sum(m))

    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    cx_a = np.sum(dx * np.arange(X)).astype(int)
    cy_a = np.sum(dy * np.arange(Y)).astype(int)
 
    if cy_a < 86:
        cy_a = 86
    img_crop = img[:, :, cy_a-cy_o : cy_a-cy_o+176]

    return img_crop

def crop_adni_to_oasis(adni_3d):
    img = np.rot90(adni_3d, k=1, axes=(0,1))
    img_crop=crop_adni3D_trav(img)
    img_crop = np.rot90(img_crop, k=1, axes=(0,2))
    img_crop=crop_adni3D_cort(img_crop)
    img_crop = np.rot90(img_crop, k=1, axes=(0,2))
    img_crop = np.rot90(img_crop, k=3, axes=(0,1))
    img_crop = np.rot90(img_crop, k=1, axes=(1,2))
    return img_crop

def get_slices_ADNI_new(IDs, N=0, d=1, dim=0, m=95, normalize=True):
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
    imgs = []
    for path in tqdm(IDs):
        img = get_ADNI_dataobj(path)
        if img == False:
            continue
        img = np.asarray(img.dataobj)
        img = crop_adni_to_oasis(img)
        imgs.append(img.take(m, axis=dim))
        for i in range(1,N+1):
            imgs.append(img.take(m+d*i, axis=dim))
            imgs.append(img.take(m-d*i, axis=dim))
    imgs = np.array(imgs) 
    if normalize:
        imgs = imgs/imgs.max()
    return imgs

def segment(X):
    hmrf = TissueClassifierHMRF(verbose=False)
    initial_segmentation, X, PVE = hmrf.classify(X, 3, 0.1)
    return np.stack([(X==1).astype(int), (X==2).astype(int), (X==3).astype(int)], axis=-1)