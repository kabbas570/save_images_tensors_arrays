import numpy as np
a1 = np.zeros([55,6,10])
print(a1.shape)

a2 = np.moveaxis(a1,-1,0)  ### move last dimenssion to first 
print(a2.shape)

a3 = np.moveaxis(a1,0,-1)  ### move first dimenssion to last 
print(a3.shape)
a2 = np.moveaxis(a1, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
