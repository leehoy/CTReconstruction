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

# R.Filtering()
R.backward()
R.SaveRecon('Recon_SheppLogan256_90.dat')
R.image = np.zeros(params['NumberOfImage'], dtype=np.float32)
eps = 1e-5
norm1 = Reconstruction(params)
norm1.proj = np.ones(
    [params['NumberOfViews'], params['NumberOfDetectorPixels'][1], params['NumberOfDetectorPixels'][0]],
    dtype=np.float32)
norm1.backward()
norm2 = Reconstruction(params)
norm2.image = np.ones(params['NumberOfImage'])
norm2.forward()
iter = 10
alpha = 1
rmse = np.zeros(iter, dtype=np.float32)
# plt.imshow(R.image[:, :, 128], cmap='gray')
# plt.show()
for i in range(iter):
    recon_tmp = R.image
    log.info('iter: %d' % i)
    R.forward()
    # plt.imshow(R.proj[40, :, :], cmap='gray')
    # plt.show()
    R.proj = (R.proj - proj0) / (norm2.proj + eps)
    # recon_tmp = R.image
    R.backward()
    # plt.imshow(R.image[:, :, 128], cmap='gray')
    # plt.show()
    R.image = recon_tmp - alpha * (R.image / norm1.image + eps)
    # plt.imshow(R.image[:, :, 128], cmap='gray')
    # plt.show()
    rmse[i] = np.sqrt(np.mean((R.image - ph) ** 2))
    log.debug(rmse[i])
R.SaveRecon('SheppLogan90_sart.dat')
