

import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread(r"C:\My_Data\KeenAI\Data\originals\img_1.JPG")/255
mask=cv2.imread(r"C:\My_Data\KeenAI\Data\originals\img_1.png")/255


def blend(image1, image2, ratio):
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha


    #image2[0,:,:] =image2*0
    image2 =image2* [1, .7, 0]
    image = image1 * alpha + image2 * beta
    return image


gt=blend(img,mask,0.7)

plt.imsave(r'C:\My_Data\KeenAI\Data\originals/' + str(3) + '.png',gt)
 
