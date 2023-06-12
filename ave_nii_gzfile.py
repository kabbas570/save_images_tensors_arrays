import nibabel as nib
import os
from skimage.morphology import skeletonize
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import SimpleITK as sitk



path = r"C:\My_Data\SEG.A. 2023\F1\training\R4\R4_gt.nii.gz"
img1 = sitk.ReadImage(path)
img1 = sitk.GetArrayFromImage(img1)

img1 = (img1 > 0.5)*0.7

img1 = img1[1,:]

img1 = np.moveaxis(img1, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
to_format_img = nib.Nifti1Image(img1, np.eye(4))  
to_format_img.set_data_dtype(np.uint8)
to_format_img.to_filename(os.path.join(r'C:\My_Data\SEG.A. 2023\sing','img_b'+'.nii.gz'))
