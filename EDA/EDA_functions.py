import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

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


def img_mean(images, box=None):
    '''Plots mean image of multi-picture numpy array'''
    fig, axes = plt.subplots(1,1, figsize=(4,6))
    plt.imshow(np.mean(images, axis=0))
    if box != None:
        axes.add_patch(Rectangle(xy=(box[0],box[2]), width=box[1]-box[0], height=box[3]-box[2], fill=False))

def compare_two_means(images1, images2, title1=None, title2=None, all=False, box=None):
    if all == False:
        fig, axes = plt.subplots(1,2, figsize=(6,8))
        axes[0].imshow(np.mean(images1, axis=0))
        axes[1].imshow(np.mean(images2, axis=0))
        axes[0].set_title(title1)
        axes[1].set_title(title2)
        if box != None:
            axes[0].add_patch(Rectangle(xy=(box[0],box[2]), width=box[1]-box[0], height=box[3]-box[2], fill=False))
            axes[1].add_patch(Rectangle(xy=(box[0],box[2]), width=box[1]-box[0], height=box[3]-box[2], fill=False))
    elif all == True:
        fig, axes = plt.subplots(3,2, figsize=(6,8))
        axes[0,0].imshow(np.mean(images1[0], axis=0))
        axes[0,1].imshow(np.mean(images2[0], axis=0))
        axes[1,0].imshow(np.mean(images1[1], axis=0))
        axes[1,1].imshow(np.mean(images2[1], axis=0))
        axes[2,0].imshow(np.mean(images1[2], axis=0))
        axes[2,1].imshow(np.mean(images2[2], axis=0))
        if box != None:
            axes[0,0].add_patch(Rectangle(xy=(box[0][0],box[0][2]), width=box[0][1]-box[0][0], height=box[0][3]-box[0][2], fill=False))
            axes[0,1].add_patch(Rectangle(xy=(box[0][0],box[0][2]), width=box[0][1]-box[0][0], height=box[0][3]-box[0][2], fill=False))
            axes[1,0].add_patch(Rectangle(xy=(box[1][0],box[1][2]), width=box[1][1]-box[1][0], height=box[1][3]-box[1][2], fill=False))
            axes[1,1].add_patch(Rectangle(xy=(box[1][0],box[1][2]), width=box[1][1]-box[1][0], height=box[1][3]-box[1][2], fill=False))
            axes[2,0].add_patch(Rectangle(xy=(box[2][0],box[2][2]), width=box[2][1]-box[2][0], height=box[2][3]-box[2][2], fill=False))
            axes[2,1].add_patch(Rectangle(xy=(box[2][0],box[2][2]), width=box[2][1]-box[2][0], height=box[2][3]-box[2][2], fill=False))
        axes[0,0].set_title(title1)
        axes[0,1].set_title(title2)
        for a in axes:
            for ax in a:
                ax.axis('off')

    
def demented(df):
    '''returns data entries where CDR >0'''
    return df[df["CDR"]>0]

def non_demented(df):
    '''returns data entries where CDR is 0'''
    return df[df["CDR"]==0]

def young(df):
    '''returns data entries where age is <=30'''
    return df[df["Age"]<=30]

def middleaged(df):
    '''returns data entries where age is >30 and <=65'''
    return df[(df["Age"]>30) & (df["Age"]<=65)]

def old(df):
    '''returns data entries where age is >65'''
    return df[df["Age"]>65]

def extract_box(image, box):
    '''Extracts from an 2D NumPy image the region specified by a box in the format [x_min, x_max, y_min, y_max]'''
    return image[box[0]:box[1],box[2]:box[3]]

