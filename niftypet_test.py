# minimal test of NiftyPET's mMR fwd/back projectors - run with NiftyPET v2.0.0
import sys
import os
import numpy as np
from time import time

from niftypet import nipet

#------------------------------------------------------------
Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()
mMRpars = {'Cnt': Cnt, 'txLUT': txLUT, 'axLUT': axLUT}
#------------------------------------------------------------

img = np.ones((127, 344, 344), dtype = np.float32)

n = 10
t_fwd = np.zeros(n)
t_back = np.zeros(n)

# extra fwd projection, since first CUDA call is always slower
tmp = nipet.frwd_prj(img, mMRpars)

for i in range(n):
    t0 = time()
    img_fwd = nipet.frwd_prj(img, mMRpars)
    t1 = time()
    test_sino = np.ones_like(img_fwd)
    t2 = time()
    ones_back = nipet.back_prj(test_sino, mMRpars)
    t3 = time()

    t_fwd[i] = t1-t0
    t_back[i] = t3-t2

print(f'{t_fwd.mean():.3f} +- {t_fwd.std():.3f}')
print(f'{t_back.mean():.3f} +- {t_back.std():.3f}')
