# Project description:  
In this project a neural network was used to predict Alzheimer's disease (AD) based on MRI images alone. The 3D images were derived from two different datasets (OASIS1 and ADNI1) and the two datasets were combined to have an almost equal distribution of healthy and diseased subjects. Analysis of the two dataset confirmed that the prevalence of AD increases with age. For the modeling, three 2D image slices representing three different axes were used and fed into a pre-trained VGG16 neural network. The outputs of the VGG16 were further been used in a DNN to make the predictions. With this model the accuracy is 76% and the recall at 94%.

![Project overview](https://github.com/tamap94/ds-capstone-alzheimers-/blob/main/figures/capstone_image_TalentApp.png)

### Dataset:   
OASIS1: 2D and 3D brain scans from 436 MRI sessions with 416 subjects 
https://www.oasis-brains.org/#data

ADNI1: 3D MRI images of 826 subjects. 
https://adni.loni.usc.edu

Tadpole challenge: From this challenge we took the numerical data for the ADNI1 subjects. 
https://tadpole.grand-challenge.org

### Goal:  
The goal was to develop a robust model which is able to predict Alzheimers from raw MRI images, which do not need extensive pre-processing. 


## Setup and requirements

```python
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To run
```python
#Load the data, preprocess it and save it in preprocessing/processed_data as segmented_slices.npz
python preprocessing/preprocess_data.py
#Train the model and save it in models
python modelling/CNN_train.py
#Model evaluation is in modelling/error_analysis.ipynb
```
