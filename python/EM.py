from Reconstruction import Reconstruction
import numpy as np
import glob, sys, os
import logging

pi = np.pi
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
# data = np.fromfile('Shepp_Logal_3d_256.dat', dtype=np.float32).reshape([256, 256, 256])
params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
          'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.5, 0.5],
          'NumberOfViews': 180, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
          'PhantomCenter': [0, 0, 0], 'Origin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'hann', 'cutoff': 1,
          'GPU': 1}
R = Reconstruction(params)
filename = 'Shepp_Logan_3d_256.dat'
# ph = np.fromfile(filename, dtype=np.float32).reshape([256, 256, 256])
R.LoadRecon(filename, params['NumberOfImage'])
ph = R.image
R.forward()
# R.proj.tofile('proj_SheppLogan256_180.dat', sep='', format='')

R.image = np.ones(params['NumberOfImage'], dtype=np.float32)
eps = 1e-5
norm = Reconstruction(params)
norm.proj = np.ones([params['NumberOfViews'], params['NumberOfDetectorPixels'][1], params['NumberOfDetectorPixels'][0]],
                    dtype=np.float32)
norm.backward()
iter = 10
rmse = np.zeros(iter, dtype=np.float32)
proj_original = R.proj
for i in range(iter):
    proj0 = proj_original
    log.info('iter: %d' % i)
    R.forward()
    # proj_tmp = R.proj
    proj0[np.where(R.proj == 0)] = 0
    R.proj[np.where(R.proj == 0)] = eps
    R.proj = proj0 / R.proj
    recon_tmp = R.image
    R.backward()
    R.image = recon_tmp * (R.image / norm.image)
    rmse[i] = np.sqrt(np.mean((R.image - ph) ** 2))
    log.debug(rmse[i])
R.SaveRecon('SheppLogan90_em.dat')
