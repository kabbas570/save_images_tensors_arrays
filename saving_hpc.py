target_dir1  = '/data/scratch/acw676/Seg_A/vae_viz/'
rv_es = out_LA_ES_RV[0,0,:].cpu().numpy()
print(rv_es.shape)
filename_gt = os.path.join(target_dir1+str(batch_idx))
np.save(filename_gt, rv_es)
            
 print(out_LA_ES_RV.shape)
 filepath = os.path.join(target_dir1, str(batch_idx)+'_pre_ES_RV.png')
 torchvision.utils.save_image(out_LA_ES_RV[0,0,:,:], filepath)

              ###  saving the gts  
to_format_img = nib.Nifti1Image(gt_single, np.eye(4))  
to_format_img.set_data_dtype(np.uint8)
to_format_img.to_filename(os.path.join(pre_apth,name[0]+'_'+str(i)+'_gt'+'.nii.gz'))
