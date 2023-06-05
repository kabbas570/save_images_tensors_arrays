import numpy as np
a1 = np.zeros([55,6,10])
print(a1.shape)

a = np.moveaxis(a1,-1,0)  ### move last dimenssion to first 
print(a.shape)
