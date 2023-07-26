import os
import SimpleITK as sitk
import numpy as np

import glob

prob_id_orig = []
for infile in sorted(glob.glob(r"C:\Users\Abbas Khan\Downloads\nii_gz/*.nii.gz")): 
    prob_id_orig.append(infile)
    

save_path = r'C:\Users\Abbas Khan\Downloads\four/'
temp = r"C:\Users\Abbas Khan\Downloads\nii_gz/"

for i in range(4):
    img = sitk.ReadImage(prob_id_orig[i])
    name = prob_id_orig[i][len(temp):-7]
    name = name+"_0000" + ".nii.gz"
    sitk.WriteImage(img,save_path+name)
