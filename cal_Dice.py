import numpy as np
from typing import Union
import SimpleITK as sitk
import pickle
import  torch
import glob

DS_org = 0
DS_new = 0
    
prob_id_orig = []
for infile in sorted(glob.glob(r"C:\My_Data\SEG.A. 2023\All_Folds\Fold1\fold_1_mydice\validation/*.nrrd")): 
    prob_id_orig.append(infile)
    
num = "C:\My_Data\SEG.A. 2023\All_Folds\Fold0\validation/"
for i in range(12):
    pred_org = sitk.ReadImage(prob_id_orig[i])
    pred_org = sitk.GetArrayFromImage(pred_org)
    
    name = name = prob_id_orig[i][len(num)+1:]
    
    gt_path = r'C:\My_Data\SEG.A. 2023\nnunet_raw\labelsTr/'+name
    gt = sitk.ReadImage(gt_path)
    gt = sitk.GetArrayFromImage(gt)
    
    
    pred_graph_path = r'C:\My_Data\SEG.A. 2023\nnresults\f_0_results/'+name
    pred_graph = sitk.ReadImage(pred_graph_path)
    pred_graph = sitk.GetArrayFromImage(pred_graph)
    
    
    single_org = (2 * (pred_org * gt).sum()) / (
                        (pred_org + gt).sum() + 1e-8)
        
        
    single_graph = (2 * (pred_graph * gt).sum()) / (
                        (pred_graph + gt).sum() + 1e-8)
    
    
    print(single_org, '  ', single_graph)
    
    DS_org+=single_org
    DS_new+=single_graph
    
    
print(DS_org/12)
print(DS_new/12)

    
