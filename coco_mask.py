from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

coco = '/content/drive/MyDrive/gen_mask_from_json/json_file/result.json'
coco=COCO(coco)
save_path = '/content/drive/MyDrive/gen_mask_from_json/gen_masks_2/'
cat_ids = coco.getCatIds()

#print(cat_ids)
for i in range(50):
    image_id = i
    img = coco.imgs[image_id]
    print(img['file_name'][32:-4])

    anns_img = np.zeros((img['height'],img['width']))

    for k in range(9):
      anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids[k], iscrowd=None)
      anns = coco.loadAnns(anns_ids)
      for ann in anns:
        anns_img1 = coco.annToMask(ann)*ann['category_id']
        anns_img  = anns_img + anns_img1

    filename = os.path.join(save_path,str(img['file_name'][32:-4]))
    np.save(filename, anns_img)
