img = data.astype(np.uint8)

target_dir1  = '/data/scratch/acw676/Seg_A/vae_viz/'
rv_es = out_LA_ES_RV[0,0,:].cpu().numpy()
print(rv_es.shape)
filename_gt = os.path.join(target_dir1+str(batch_idx))
np.save(filename_gt, rv_es)
            
 print(out_LA_ES_RV.shape)
 filepath = os.path.join(target_dir1, str(batch_idx)+'_pre_ES_RV.png')
 torchvision.utils.save_image(out_LA_ES_RV[0,0,:,:], filepath)

import nibabel as nib
import os

              ###  saving the gts  
to_format_img = nib.Nifti1Image(gt_single, np.eye(4))  
to_format_img.set_data_dtype(np.uint8)
to_format_img.to_filename(os.path.join(pre_apth,name[0]+'_'+str(i)+'_gt'+'.nii.gz'))


 #plt.imsave('D:\\Greg\\Research\\Code\\abbas\\001\\001_Orig_sa' + str(i) + '.png', temp[i, :, :])
            
                                    #### Saving the Visual FM and Reuslts  #####
            name = name[0]
            out_LA_ES_LV = out_LA_ES_LV[0,0,:].float()
            out_LA_ES_MYO = out_LA_ES_MYO[0,0,:].float()
            out_LA_ES_RV = out_LA_ES_RV[0,0,:].float()

            filepath = os.path.join(target_dir1, str(name)+'_LV'+'.png')
            torchvision.utils.save_image(out_LA_ES_LV, filepath)
            
            filepath = os.path.join(target_dir1, str(name)+'_MYO'+'.png')
            torchvision.utils.save_image(out_LA_ES_MYO, filepath)
            
            filepath = os.path.join(target_dir1, str(name)+'_RV'+'.png')
            torchvision.utils.save_image(out_LA_ES_RV, filepath)
