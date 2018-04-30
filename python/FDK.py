from Reconstruction import Reconstruction
import numpy as np
import glob, sys, os
import logging
import matplotlib.pyplot as plt

pi = np.pi
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
# data = np.fromfile('Shepp_Logal_3d_256.dat', dtype=np.float32).reshape([256, 256, 256])
params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
          'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.5, 0.5],
          'NumberOfViews': 90, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
          'PhantomCenter': [0, 0, 0], 'Origin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'hann', 'cutoff': 1,
          'GPU': 1}
R = Reconstruction(params)
filename = 'Shepp_Logan_3d_256.dat'

R.LoadRecon(filename, params['NumberOfImage'])
ph = R.image
R.forward()
print(R.proj.shape)
proj0 = R.proj
R.SaveProj('proj_SheppLogan256_90.dat')

R.Filtering()
R.backward()
R.SaveRecon('Recon_SheppLogan256_90_fdk.dat')

