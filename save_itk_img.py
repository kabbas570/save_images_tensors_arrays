path_imgs = r'C:\My_Data\SEG.A. 2023\1_patches\imgs/'
path_gts = r'C:\My_Data\SEG.A. 2023\1_patches\gts/'
gt = a1[1][0,k,:].numpy()
gt = sitk.GetImageFromArray(gt)
gt.SetSpacing([0.707,0.707,2.0])
sitk.WriteImage(gt,path_gts+name+'.nii.gz')
