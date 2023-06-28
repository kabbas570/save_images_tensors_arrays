from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
           ###########  Dataloader  #############
import numpy as np
import SimpleITK as sitk
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
depth = 64 

def resample_image(image: sitk.Image, out_spacing: Tuple[float] = (0.707, .707, 2.0),
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            out_size = (np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
        else:
            out_size = np.array(out_size)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(image)
    
gt = sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\one\D4\D4.seg.nrrd")
gt = resample_image(gt,is_label=True) 
gt = sitk.GetArrayFromImage(gt)



p1 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\gts\D4_0.nii.gz"))
p2= sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\gts\D4_1.nii.gz"))
p3 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\gts\D4_2.nii.gz"))
p4 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\gts\D4_3.nii.gz"))
p5 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\gts\D4_4.nii.gz"))


# p1 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\zeros\img\D4_0.nii.gz"))
# p2= sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\zeros\img\D4_0.nii.gz"))
# p3 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\zeros\img\D4_0.nii.gz"))
# p4 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\zeros\img\D4_0.nii.gz"))
# p5 = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\zeros\img\D4_0.nii.gz"))

c = np.concatenate((p1,p2,p3,p4,p5),axis=0)
#c = np.moveaxis(c,-1,0)  ### move first dimenssion to last 



def Back_to_orgSize_new(img,org_shape):  ## this accepts mumpyarray of size [dEPTH,h,w]
    temp = np.zeros(org_shape)
    del_features = img.shape[0] - org_shape[0]
    start = (org_shape[0]//64)*64    
    end = start + del_features
    d1 = int((org_shape[1]-DIM_)/2)
    d2 = int((org_shape[2]-DIM_)/2)
    n1 = int(org_shape[1]-d1)
    n2 = int(org_shape[2]-d2)
    
    if( org_shape[1]%2)!=0:
        n1=n1-1
    if( org_shape[2]%2)!=0:
        n2=n2-1
        
    i1 = img[0:start,:]
    i2 = img[end:img.shape[0],:]

    img_new = np.concatenate((i1,i2), axis=0)    
    temp[:,d1:n1,d2:n2] = img_new
    
    #temp[:,d1:n1,d2:n2] = img[0:img.shape[0]-del_features,:]
    
    return temp


def Back_to_orgSize(img,org_shape):  ## this accepts mumpyarray of size [dEPTH,h,w]
    temp = np.zeros(org_shape)
    del_features = img.shape[0] - org_shape[0]

    d1 = int((org_shape[1]-DIM_)/2)
    d2 = int((org_shape[2]-DIM_)/2)
    n1 = int(org_shape[1]-d1)
    n2 = int(org_shape[2]-d2)
    
    if( org_shape[1]%2)!=0:
        n1=n1-1
    if( org_shape[2]%2)!=0:
        n2=n2-1
    
    temp[:,d1:n1,d2:n2] = img[0:img.shape[0]-del_features,:]
    return temp

org_shape= gt.shape

back = Back_to_orgSize_new(c,org_shape)
#back = sitk.GetImageFromArray(back)
#gt_itk = sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\one\D4\D4.seg.nrrd")
#back.CopyInformation(gt_itk)
#t = gt.astype(np.float64)

single_ = (2 * (gt * back).sum()) / (
               (gt + back).sum() + 1e-8)

print(single_)

# import nibabel as nib

# import numpy as np
# img1 = np.moveaxis(back, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
# to_format_img = nib.Nifti1Image(img1, np.eye(4))  
# to_format_img.set_data_dtype(np.uint8)
# to_format_img.to_filename(os.path.join(r'C:\My_Data\SEG.A. 2023','back'+'.nii.gz'))


plt.figure()
plt.imshow(gt[50,:])

img = sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\imgs\D4_3new.nii.gz")
print(np.array(img.GetSpacing()))
img = sitk.GetArrayFromImage(img)

org = sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\1_patches\imgs\D4_3new.nii.gz")
print(np.array(org.GetSpacing()))
org = sitk.GetArrayFromImage(org)


pre = sitk.ReadImage(r"C:\My_Data\SEG.A. 2023\one\D4\D4.seg.nrrd")
print(np.array(pre.GetSpacing()))
pre = sitk.GetArrayFromImage(pre)

for i in range(55,64):
    plt.figure()
    plt.imshow(img[:,:,i])
    plt.figure()
    plt.imshow(pre[:,:,i])



img[np.where(pre==1)]=10

for i in range(55,64):
    plt.figure()
    plt.imshow(img[i,:])
    
    
path_gts = r'C:\My_Data\SEG.A. 2023/'
img = sitk.GetImageFromArray(gt)
img.SetSpacing([0.707,0.707,2.0])
sitk.WriteImage(img,path_gts+'gt'+'.nii.gz')
        
