import cc3d
import numpy as np
import SimpleITK as sitk

import glob
from medpy import metric


def calculate_metric_percase(pred, gt):
      dice = metric.binary.dc(pred, gt)
      hd = metric.binary.hd95(pred, gt)
      return dice, hd
  
    

# labels_in_itk = sitk.ReadImage(r"C:\My_Data\qmul\asad\test_1.nii.gz")
# print(np.array(labels_in_itk.GetSpacing()))
# labels_in = sitk.GetArrayFromImage(labels_in_itk)
# labels_out = cc3d.dust(
#   labels_in, threshold=500, 
#   connectivity=26, in_place=False
# )
# a = np.zeros([50,50])
# a[10:20,10:20]=1
# a[30:50,20:40]=1
# a[10:20,30:40]=1
# a[2:8,30:32]=1
# labels_out, N = cc3d.largest_k(
#   a, k=5, 
#   connectivity=26, delta=0,
#   return_N=True,
# )
# print(N)
# labels_out, N = cc3d.largest_k(
#   labels_in, k=1, 
#   connectivity=26, delta=0,
#   return_N=True,
# )
# print(N)
# labels_out = sitk.GetImageFromArray(labels_out)
# labels_out.CopyInformation(labels_in_itk)
# print(np.array(labels_out.GetSpacing()))
# sitk.WriteImage(labels_out, r'C:\My_Data\qmul\asad/'+'test_2.nii.gz', True)

 
prob_id_orig = []
for infile in sorted(glob.glob(r"C:\My_Data\qmul\asad\lowres_final/*.nrrd")): 
    prob_id_orig.append(infile)
    

temp = "C:\My_Data\qmul\asad\lowres_final/"

Dice_org =0
Dice_new = 0
HD_new  =0
HD_old = 0
for i in range(2):

    pre_org = sitk.ReadImage(prob_id_orig[i])
    pre_org = sitk.GetArrayFromImage(pre_org)
    
    name = prob_id_orig[i][len(temp):]
    
    gt = sitk.ReadImage(r'C:\My_Data\SEG.A. 2023\nnunet_raw\labelsTr'+name)
    gt = sitk.GetArrayFromImage(gt)
    
    
    
    # pre_cc , _ = cc3d.largest_k(
    #   pre_org, k=1, 
    #   connectivity=26, delta=0,
    #   return_N=True,
    # )
    
    pre_cc = cc3d.dust(
      pre_org, threshold=100, 
      connectivity=6, in_place=False
    )
    
    
    single_cc,single_hd_cc = calculate_metric_percase(pre_cc,gt)
        
        
    single_org,single_hd_org = calculate_metric_percase(pre_org,gt)
    
    
    print(single_org, '  ', single_cc)
    print(single_hd_org, '  ', single_hd_cc)
    
    Dice_org+=single_org
    
    Dice_new+=single_cc
    
    HD_new+=single_hd_cc
    HD_old+=single_hd_org
    
  
print('overall_scre ')
print(Dice_org/2, '   ',Dice_new/2 )
print(HD_old/2, '   ',HD_new/2 )
    
    
    
    
    



    
    
    
    
    
    
