
import numpy as np
from typing import Union
import SimpleITK as sitk
import pickle
import  torch
import glob
from medpy import metric


def calculate_metric_percase(pred, gt):
      dice = metric.binary.dc(pred, gt)
      hd = metric.binary.hd95(pred, gt)
      return dice, hd

p1 = r"C:\My_Data\qmul\test_3.nrrd"
p2 = r"C:\My_Data\SEG.A. 2023\All_Folds\Fold0\validation\SEGA_030_K20.nrrd"
gt = r"C:\My_Data\SEG.A. 2023\All_Folds\Fold0\val\K20\K20.seg.nrrd"




gt = sitk.ReadImage(gt)
gt = sitk.GetArrayFromImage(gt)

p1 = sitk.ReadImage(p1)
p1 = sitk.GetArrayFromImage(p1)

p2 = sitk.ReadImage(p2)
p2 = sitk.GetArrayFromImage(p2)



single_org,single_hd_org = calculate_metric_percase(p1,gt)

print(single_org,'   ',single_hd_org )
single_org,single_hd_org = calculate_metric_percase(p2,gt)
print(single_org,single_hd_org )



import SimpleITK



input_image=SimpleITK.ReadImage(r"C:\My_Data\qmul\my\ct\A1.nrrd")
input_image = SimpleITK.GetArrayFromImage(input_image)


input_image  =input_image[0:72,0:160,0:160]

input_image = SimpleITK.GetImageFromArray(input_image)

SimpleITK.WriteImage(input_image,r'C:\My_Data\qmul\my\ct/'+'A1.nrrd')
