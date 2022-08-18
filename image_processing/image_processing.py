import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

## image loading
def load_MRI_gifs(IDs, plane="all"):
    '''Given a set of IDs ("OAS1_0xxx_MR1") returns the atlas corrected traverse, sagittal and cortical images as numpy arrays'''
    tra = []
    sag = []
    cor = []
    for path in IDs:
        path1 = '../data/Oasis_Data/' + path + '/PROCESSED/MPRAGE/T88_111/'
        for img in os.listdir(path1):
            if img.endswith('t88_gfc_tra_90.gif'):
                tra.append(plt.imread(path1+img))
            elif img.endswith('t88_gfc_sag_95.gif'):
                sag.append(plt.imread(path1+img))
            elif img.endswith('t88_gfc_cor_110.gif'):
                cor.append(plt.imread(path1+img))
    if plane == "traverse":
        return np.array(tra)
    elif plane == "sagittal":
        return np.array(sag)
    elif plane == "cortical":
        return np.array(cor)
    elif plane == "all":
        return [np.array(tra), np.array(sag), np.array(cor)]
    else:
        print("Inncorect selection of planes")


## region extraction
def extract_box(image, box):
    '''Extracts from an 2D NumPy image the region specified by a box in the format [x_min, x_max, y_min, y_max]'''
    return image[box[0]:box[1],box[2]:box[3]]

def stacked_boxes(img_stack, box):
    '''Cuts a specified box from a 3D stack of images'''
    out=[]
    for img in img_stack:
        #np.append(out, extract_box(img_stack[i], box), axis=0)
        out.append(extract_box(img, box))
    return np.array(out)

## slice extraction