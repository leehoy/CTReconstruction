# from Reconstruction import Reconstruction
from Reconstruction_detector_change import Reconstruction
import numpy as np
import glob, sys, os
import logging
import time
import matplotlib.pyplot as plt

pi = np.pi
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
# data = np.fromfile('Shepp_Logal_3d_256.dat', dtype=np.float32).reshape([256, 256, 256])
params = {'SourceInit': [0, 500.0, -20], 'DetectorInit': [0, -500.0, -20], 'StartAngle': 0,
          'EndAngle': 4 * pi, 'NumberOfDetectorPixels': [512, 16], 'DetectorPixelSize': [1.2, 1.0],
          'NumberOfViews': 1440, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
          'PhantomCenter': [0, 0, 0], 'Origin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'hann', 'cutoff': 1,
          'GPU': 1, 'DetectorShape': 'Curved', 'Pitch': 0.6}
R = Reconstruction(params)
filename = 'Shepp_Logan_3d_256.dat'
# filename = 'Shepp_Logan_3d_256.dat'

R.LoadRecon(filename, params['NumberOfImage'])
ph = np.copy(R.image)
start_time = time.time()
R.forward()
log.info('Forward %f' % (time.time() - start_time))
print(R.proj.shape)
# proj0 = R.proj
R.SaveProj('HelicalProj_SheppLogan_2880_16.dat')
print(time.time() - start_time)
# start_time = time.time()
# R.Filtering()
# R.backward()
# log.info('Backward: %f' % (time.time() - start_time))
# R.SaveRecon('Recon_SheppLogan256_90_fdk.dat')
