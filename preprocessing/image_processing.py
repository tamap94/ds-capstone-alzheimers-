import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from dipy.segment.tissue import TissueClassifierHMRF

## image loading
def load_MRI_gifs(IDs, plane="all", rescale=True):
    '''Given a set of IDs ("OAS1_0xxx_MR1") returns the atlas corrected traverse, sagittal and cortical images as numpy arrays'''
    tra = []
    sag = []
    cor = []
    mask = []
    for path in IDs:
        path1 = '../data/Oasis_Data/' + path + '/PROCESSED/MPRAGE/T88_111/'
        for img in os.listdir(path1):
            if img.endswith('t88_gfc_tra_90.gif'):
                tra.append(plt.imread(path1+img))
            elif img.endswith('t88_gfc_sag_95.gif'):
                sag.append(plt.imread(path1+img))
            elif img.endswith('t88_gfc_cor_110.gif'):
                cor.append(plt.imread(path1+img))
            elif img.endswith('masked_gfc_tra_90.gif'):
                mask.append(plt.imread(path1+img))
    if plane == "traverse":
        return np.array(tra)
    elif plane == "sagittal":
        return np.array(sag)
    elif plane == "cortical":
        return np.array(cor)
    elif plane == "mask":
        return np.array(mask)
    elif plane == "all":
        return [np.array(tra), np.array(sag), np.array(cor)]
    else:
        print("Inncorect selection of planes")


## region extraction
def extract_box(image, box):
    '''Extracts from an 2D NumPy image the region specified by a box in the format [x_min, x_max, y_min, y_max]'''
    return image[box[2]:box[3],box[0]:box[1]]

def stacked_boxes(img_stack, box):
    '''Cuts a specified box from a 3D stack of images'''
    out=[]
    for img in img_stack:
        #np.append(out, extract_box(img_stack[i], box), axis=0)
        out.append(extract_box(img, box))
    return np.array(out)

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

def segment(X):
    hmrf = TissueClassifierHMRF(verbose=False)
    initial_segmentation, X, PVE = hmrf.classify(X, 3, 0.1)
    return np.stack([(X==1).astype(int), (X==2).astype(int), (X==3).astype(int)], axis=-1)