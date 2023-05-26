target_dir1  = '/data/scratch/acw676/Seg_A/vae_viz/'
rv_es = out_LA_ES_RV[0,0,:].cpu().numpy()
print(rv_es.shape)
filename_gt = os.path.join(target_dir1+str(batch_idx))
np.save(filename_gt, rv_es)
            
 print(out_LA_ES_RV.shape)
 filepath = os.path.join(target_dir1, str(batch_idx)+'_pre_ES_RV.png')
 torchvision.utils.save_image(out_LA_ES_RV[0,0,:,:], filepath)
