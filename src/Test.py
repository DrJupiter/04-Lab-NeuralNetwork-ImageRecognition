import numpy as np
#print(np.arange(1,5))

import torch

#print(torch.tensor(2))

import matplotlib
from matplotlib import pyplot as plt
import skimage
import imageio




Ja =1
nej=0
T=np.array([1,1,1,0,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1])
print(np.count_nonzero(T)/80)

amount = np.array([np.count_nonzero(T),len(T)-np.count_nonzero(T)])
label = np.array(["correct","not_correct"])

print(np.mean(T)*100)


stan1 = T
stan0 = T

error = [np.std(stan1),np.std(stan0)]

plt.bar(label,amount,color=["green","red"])

plt.show()
