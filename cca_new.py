import cc3d
import numpy as np
import SimpleITK as sitk

import glob
from medpy import metric


def calculate_metric_percase(pred, gt):
      dice = metric.binary.dc(pred, gt)
      #hd = metric.binary.hd95(pred, gt)
      return dice, dice
  
 
prob_id_orig = []
for infile in sorted(glob.glob(r"C:\My_Data\qmul\asad\lowres_final/*.nrrd")): 
    prob_id_orig.append(infile)
    

temp = "C:\My_Data\qmul\asad\lowres_final/"

Dice_org =0
Dice_new = 0
HD_new  =0
HD_old = 0
for i in range(50,56):

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
      pre_org, threshold=50, 
      connectivity=18, in_place=False
    )
    
    
    single_cc,single_hd_cc = calculate_metric_percase(pre_cc,gt)
        
        
    single_org,single_hd_org = calculate_metric_percase(pre_org,gt)
    
    
    print(single_org, '  ', single_cc)
    #print(single_hd_org, '  ', single_hd_cc)
    
    Dice_org+=single_org
    
    Dice_new+=single_cc
    
    HD_new+=single_hd_cc
    HD_old+=single_hd_org
    
  
print('overall_scre ')
print(Dice_org/2, '   ',Dice_new/2 )
#print(HD_old/2, '   ',HD_new/2 )
    
    
    
    
    



    
    
    
    
    
    
