import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import cv2
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import torchio as tio

DIM_ = 256

def generate_label_4(gt):
        temp_ = np.zeros([4,gt.shape[1],DIM_,DIM_])
        temp_[0:1,:,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:,:][np.where(gt==0)]=1
        return temp_

def resample_image(image: sitk.Image ,
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
                
        out_spacing = (1.25, 1.25,original_spacing[2])

        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            #out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
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

def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_LA_ES(img):
        img = (img-114.8071)/191.2891
        return img 
    
img = sitk.ReadImage(r"C:\My_Data\M2M Data\data\data_2\single\imgs\052\052_LA_ED.nii.gz")
img = resample_image(img,is_label=False)      ## --> [H,W,C]
img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
org_dim3 = img.shape[0]
org_dim1 = img.shape[1]
org_dim2 = img.shape[2] 
img = Normalization_LA_ES(img)
img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
img_LA_ES = np.expand_dims(img, axis=0)
        
img = sitk.ReadImage(r"C:\My_Data\M2M Data\data\data_2\single\imgs\052\052_LA_ED_gt.nii.gz")
img = resample_image(img,is_label=True)
img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
org_dim3 = img.shape[0]
org_dim1 = img.shape[1]
org_dim2 = img.shape[2] 
img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
temp_LA_ES = np.expand_dims(img, axis=0)




transforms_geometric = tio.Compose({
        #tio.RandomFlip(axes=([1,2])): .3,  ## axis [0,1] or [1,2]
        #tio.RandomFlip(axes=([0,1])): .3,  ## axis [0,1] or [1,2]
        tio.RandomAffine(degrees=(100,0,0)): 0.3, ## for 2D rotation 
})


transforms_all = tio.OneOf({
        tio.RandomAffine(degrees=(100,0,0)): 0.1, ## for 2D rotation 
        tio.RandomBiasField(): .1,  ## axis [0,1] or [1,2]
        tio.RandomGhosting(axes=([1,2])): 0.1,
        tio.RandomBlur(): 0.1,
        tio.RandomGamma(): 0.1,   
        tio.RandomNoise(mean=0.2,std=0.2):0.1,
})



d = {}
d['Image'] = tio.Image(tensor = img_LA_ES, type=tio.INTENSITY)
d['Mask'] = tio.Image(tensor = temp_LA_ES, type=tio.LABEL)
sample = tio.Subject(d)


if transforms_all is not None:
    transformed_tensor = transforms_all(sample)
    img_LA_ES1 = transformed_tensor['Image'].data
    temp_LA_ES1 = transformed_tensor['Mask'].data




plt.figure()
plt.imshow(img_LA_ES[0,0,:])

plt.figure()
plt.imshow(temp_LA_ES[0,0,:])

plt.figure()
plt.imshow(img_LA_ES1[0,0,:])

plt.figure()
plt.imshow(temp_LA_ES1[0,0,:])


