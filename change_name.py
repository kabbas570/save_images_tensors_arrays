import os
import SimpleITK as sitk
import numpy as np

import glob

prob_id_orig = []
for infile in sorted(glob.glob("/data/scratch/acw676/inp_imgs/org/*.nii.gz")): 
    prob_id_orig.append(infile)
    

save_path = '/data/scratch/acw676/inp_imgs/org/'
temp = "/data/scratch/acw676/inp_imgs/org/all_nn/"

for i in range(4):
    img = sitk.ReadImage(prob_id_orig[i])
    name = prob_id_orig[i][len(temp):-7]
    name = name+"_0000" + ".nii.gz"
    sitk.WriteImage(img,save_path+name)
