import os
import sys
import numpy as np
import numpy.matlib
from scipy.interpolate import interp2d, griddata
import glob
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy.matlib
import pycuda.gpuarray
from math import ceil
import time
from GPUFuncs import *
import logging

# define logger
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# function alias starts
sin = np.sin
cos = np.cos
atan = np.arctan
tan = np.tan
sinc = np.sinc
sqrt = np.sqrt
repmat = numpy.matlib.repmat
# ceil = np.ceil
pi = np.pi
floor = np.floor
log2 = np.log2
fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
real = np.real
# function alias ends
mod = DefineGPUFuns()


class ErrorDescription:
    def __init__(self, value):
        if (value == 1):
            self.msg = 'Unknown variables'
        elif (value == 2):
            self.msg = 'Unknown data precision'
        elif (value == 3):
            self.msg = 'Number of file is different from number of projection data required'
        elif (value == 4):
            self.msg = 'Cutoff have to be pose between 0 and 0.5'
        elif (value == 5):
            self.msg = 'Smooth have to be pose between 0 and 1'
        elif (value == 7):
            self.msg = 'Geometry dose not have proper forward projection function'
        else:
            self.msg = 'Unknown error'

    def __str__(self):
        return self.msg


class Reconstruction(object):

    def __init__(self, params):
        self.params = {'SourceInit': [0, 0, 0], 'DetectorInit': [0, 0, 0], 'StartAngle': 0,
                       'EndAngle': 0, 'NumberOfDetectorPixels': [0, 0], 'DetectorPixelSize': [0, 0],
                       'NumberOfViews': 0, 'ImagePixelSpacing': [0, 0, 0], 'NumberOfImage': [0, 0, 0],
                       'PhantomCenter': [0, 0, 0], 'Origin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'ram-lak',
                       'cutoff': 1, 'GPU': 0, 'DetectorShape': 'Flat', 'Pitch': 0}
        self.params = params

    #         start_time = time.time()
    #         self.proj = self.LoadProj(filename, [self.params['NumberOfViews'], self.params['NumberOfDetectorPixels'][1], self.params['NumberOfDetectorPixels'][0]], dtype=np.float32)
    # print('File load: ' + str(time.time() - start_time))
    def LoadProj(self, filename, image_size, dtype=np.float32):
        self.proj = np.fromfile(filename, dtype=dtype).reshape(image_size)
        # return proj

    def LoadRecon(self, filename, image_size, dtype=np.float32):
        self.image = np.fromfile(filename, dtype=dtype).reshape(image_size)
        # return image

    def SaveProj(self, filename):
        self.proj.tofile(filename, sep='', format='')

    def SaveRecon(self, filename):
        self.image.tofile(filename, sep='', format='')

    def CurvedDetectorConstruction(self, Source, DetectorCenter, SDD, angle):
        eu = [cos(angle), sin(angle), 0]
        ew = [sin(angle), -cos(angle), 0]
        ev = [0, 0, 1]
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        da = du / SDD
        u = (np.arange(0, nu) - (nu - 1) / 2.0) * da
        v = (np.arange(0, nv) - (nv - 1) / 2.0) * dv
        DetectorIndex = np.zeros([3, len(v), len(u)], dtype=np.float32)
        U, V = np.meshgrid(u, v)
        # V, U = np.meshgrid(v, u)
        DetectorIndex[0, :, :] = Source[0] + SDD * sin(U) * eu[0] + SDD * cos(U) * ew[0] - V * ev[0]
        DetectorIndex[1, :, :] = Source[1] + SDD * sin(U) * eu[1] + SDD * cos(U) * ew[1] - V * ev[1]
        DetectorIndex[2, :, :] = Source[2] + SDD * sin(U) * eu[2] + SDD * cos(U) * ew[2] - V * ev[2]
        u2 = (np.arange(0, nu + 1) - (nu - 1) / 2.0) * da - da / 2.0
        v2 = (np.arange(0, nv + 1) - (nv - 1) / 2.0) * dv - dv / 2.0
        DetectorBoundary = np.zeros([3, len(v2), len(u2)], dtype=np.float32)
        U2, V2 = np.meshgrid(u2, v2)
        DetectorBoundary[0, :, :] = Source[0] + SDD * sin(U2) * eu[0] + SDD * cos(U2) * ew[0] - V2 * ev[0]
        DetectorBoundary[1, :, :] = Source[1] + SDD * sin(U2) * eu[1] + SDD * cos(U2) * ew[1] - V2 * ev[1]
        DetectorBoundary[2, :, :] = Source[2] + SDD * sin(U2) * eu[2] + SDD * cos(U2) * ew[2] - V2 * ev[2]
        # for uu in range(len(DetectorLength[0])):
        #     for vv in range(len(DetectorLength[1])):
        #         DetectorIndex[:, uu, vv] = Source[0] + SDD * sin(DetectorLength[0][uu]) * eu + Source[1] + SDD * cos(
        #             DetectorLength[0][uu]) * ew + Source[3] + DetectorLength[1][vv] * ev

        return DetectorIndex, DetectorBoundary

    def FlatDetectorConstruction(self, Source, DetectorCenter, SDD, angle):
        tol_min = 1e-5
        tol_max = 1e6
        eu = [cos(angle), sin(angle), 0]
        ew = [sin(angle), -cos(angle), 0]
        ev = [0.0, 0.0, 1.0]
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        # dv=-1.0*dv
        u = (np.arange(0, nu) - (nu - 1.0) / 2.0) * du
        v = (np.arange(0, nv) - (nv - 1.0) / 2.0) * dv
        DetectorIndex = np.zeros([3, len(v), len(u)], dtype=np.float32)

        # U, V = np.meshgrid(u, v)
        U, V = np.meshgrid(u, v)
        DetectorIndex[0, :, :] = Source[0] + U * eu[0] + SDD * ew[0] - V * ev[0]
        DetectorIndex[1, :, :] = Source[1] + U * eu[1] + SDD * ew[1] - V * ev[1]
        DetectorIndex[2, :, :] = Source[2] + U * eu[2] + SDD * ew[2] - V * ev[2]
        u2 = (np.arange(0, nu + 1) - (nu - 1) / 2.0) * du - du / 2.0
        v2 = (np.arange(0, nv + 1) - (nv - 1) / 2.0) * dv - dv / 2.0
        DetectorBoundary = np.zeros([3, len(v2), len(u2)], dtype=np.float32)
        U2, V2 = np.meshgrid(u2, v2)
        DetectorBoundary[0, :, :] = Source[0] + U2 * eu[0] + SDD * ew[0] - V2 * ev[0]
        DetectorBoundary[1, :, :] = Source[1] + U2 * eu[1] + SDD * ew[1] - V2 * ev[1]
        DetectorBoundary[2, :, :] = Source[2] + U2 * eu[2] + SDD * ew[2] - V2 * ev[2]
        return DetectorIndex, DetectorBoundary

    @staticmethod
    def _optimalGrid(GridSize):

        if (sqrt(GridSize).is_integer()):
            gridX = int(np.sqrt(GridSize))
            gridY = gridX
        else:
            Candidates = np.arange(1, GridSize + 1)
            Division = GridSize / Candidates
            CheckInteger = Division % 1
            Divisors = Candidates[np.where(CheckInteger == 0)]
            DivisorIndex = int(len(Divisors) / 2)
            gridX = Divisors[DivisorIndex]
            gridY = Divisors[DivisorIndex - 1]
        return (int(gridX), int(gridY))

    @staticmethod
    def Filter(N, pixel_size, FilterType, cutoff):
        '''
        TO DO: Ram-Lak filter implementation
                   Argument for name of filter
        '''
        try:
            if cutoff > 1 or cutoff < 0:
                raise ErrorDescription(4)
        except ErrorDescription as e:
            print(e)
        x = np.arange(0, N) - (N - 1) / 2.0
        h = np.zeros(len(x))
        h[np.where(x == 0)] = 1 / (8 * pixel_size ** 2)
        odds = np.where(x % 2.0 == 1)
        h[odds] = -0.5 / (pi * pixel_size * x[odds]) ** 2
        h = h[0:-1]
        filter = abs(fftshift(fft(h)))
        w = 2 * pi * x[0:-1] / (N - 1)
        # print(filter.shape, w.shape)
        if FilterType == 'ram-lak':
            pass  # Do nothing
        elif FilterType == 'shepp-logan':
            zero = np.where(w == 0)
            tmp = filter[zero]
            filter = filter * sin(w / (2.0 * cutoff)) / (w / (2.0 * cutoff))
            filter[zero] = tmp * sin(w[zero] / (2.0 * cutoff))
        elif FilterType == 'cosine':
            filter = filter * cos(w / (2.0 * cutoff))
        elif FilterType == 'hamming':
            filter = filter * (0.54 + 0.46 * (cos(w / cutoff)))
        elif FilterType == 'hann':
            filter = filter * (0.5 + 0.5 * cos(w / cutoff))

        filter[np.where(abs(w) > pi * cutoff / (2.0 * pixel_size))] = 0
        return filter

    def Filtering(self):
        [du, dv] = self.params['DetectorPixelSize']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        ZeroPaddedLength = int(2.0 ** (ceil(log2(2.0 * (nu - 1)))))
        ki = (np.arange(0, nu + 1) - nu / 2.0) * du
        p = (np.arange(0, nv + 1) - nv / 2.0) * dv
        for i in range(self.proj.shape[0]):
            self.proj[i, :, :] = self.filter_proj(self.proj[i, :, :], ki, p, self.params)

    def backward(self):
        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        cutoff = self.params['cutoff']
        FilterType = self.params['FilterType']
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        Source_Init = np.array(self.params['SourceInit'])
        Detector_Init = np.array(self.params['DetectorInit'])
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = np.array(self.params['Origin'])
        PhantomCenter = np.array(self.params['PhantomCenter'])
        gpu = self.params['GPU']
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2.0))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2.0))
        # Calculates detector center
        # angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        # angle = angle[0:-1]
        # dtheta = angle[1] - angle[0]
        # deltaS = du * SAD / SDD
        # Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx + 1)) * dx
        # Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny + 1)) * dy
        # Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz + 1)) * dz
        # Xpixel = Xplane[0:-1]
        # Ypixel = Yplane[0:-1]
        # Zpixel = Zplane[0:-1]
        # ki = (np.arange(0, nu + 1) - nu / 2.0) * du
        # p = (np.arange(0, nv + 1) - nv / 2.0) * dv
        alpha = 0
        beta = 0
        gamma = 0
        eu = [cos(gamma) * cos(alpha), sin(alpha), sin(gamma)]
        ev = [cos(gamma) * -sin(alpha), cos(gamma) * cos(alpha), sin(gamma)]
        ew = [0, 0, 1]
        # print('Variable initialization: ' + str(time.time() - start_time))
        # Source = np.array([-SAD * sin(angle[0]), SAD * cos(angle[0]), 0])  # z-direction rotation
        # Detector = np.array([(SDD - SAD) * sin(angle[0]), -(SDD - SAD) * cos(angle[0]), 0])
        # DetectorLength = np.array(
        #    [np.arange(floor(-nu / 2), floor(nu / 2) + 1) * du, np.arange(floor(-nv / 2), floor(nv / 2) + 1) * dv])
        # DetectorVectors = [eu, ev, ew]
        # DetectorIndex = self.DetectorConstruction(Detector, DetectorLength, DetectorVectors, angle[0])
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        # plt.plot(ki)
        # plt.show()
        start_time = time.time()
        if (self.params['Method'] == 'Distance'):
            recon = self.distance_backproj()
        elif (self.params['Method'] == 'Ray'):
            pass
            # recon = self.ray(DetectorIndex, Source, Detector, angle)
        self.image = recon

    @staticmethod
    def filter_proj(proj, ki, p, params):
        [du, dv] = params['DetectorPixelSize']
        [nu, nv] = params['NumberOfDetectorPixels']
        ZeroPaddedLength = int(2 ** (ceil(log2(2.0 * (nu - 1)))))
        R = sqrt(np.sum((np.array(params['SourceInit']) - np.array(params['PhantomCenter'])) ** 2.0))
        D = sqrt(np.sum((np.array(params['DetectorInit']) - np.array(params['PhantomCenter'])) ** 2.0))
        # print(ki.shape, p.shape)
        [kk, pp] = np.meshgrid(ki[0:-1] * R / (R + D), p[0:-1] * R / (R + D))
        weight = R / (sqrt(R ** 2.0 + kk ** 2.0 + pp ** 2.0))

        deltaS = du * R / (R + D)
        filter = Reconstruction.Filter(
            ZeroPaddedLength + 1, du * R / (D + R), params['FilterType'], params['cutoff'])
        weightd_proj = weight * proj
        Q = np.zeros(weightd_proj.shape, dtype=np.float32)
        for k in range(nv):
            tmp = real(ifft(
                ifftshift(filter * fftshift(fft(weightd_proj[k, :], ZeroPaddedLength)))))
            Q[k, :] = tmp[0:nu] * deltaS

        return Q

    def distance_backproj(self):
        # proj should be filtered data
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        Source = self.params['SourceInit']
        Detector = self.params['DetectorInit']
        R = sqrt(np.sum((np.array(Source) - np.array(self.params['PhantomCenter'])) ** 2.0))
        rotation_vector = [0, 0, 1]
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        nViews = self.params['NumberOfViews']
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = angle[0:-1]
        PhantomCenter = self.params['PhantomCenter']
        dtheta = angle[1] - angle[0]
        Xpixel = PhantomCenter[0] + (np.arange(0, nx) - (nx - 1) / 2.0) * dx
        Ypixel = PhantomCenter[1] + (np.arange(0, ny) - (ny - 1) / 2.0) * dy
        Zpixel = PhantomCenter[2] + (np.arange(0, nz) - (nz - 1) / 2.0) * dz
        ki = (np.arange(0, nu + 1) - (nu - 1) / 2.0) * du
        p = (np.arange(0, nv + 1) - (nv - 1) / 2.0) * dv
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        if self.params['GPU']:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            TotalSize = nx * ny * nz
            if (TotalSize < MAX_THREAD_PER_BLOCK):
                blockX = nx * ny * nz
                blockY = 1
                blockZ = 1
                gridX = 1
                gridY = 1
            else:
                blockX = 32
                blockY = 32
                blockZ = 1
                GridSize = ceil(TotalSize / (blockX * blockY))
                try:
                    if (GridSize < MAX_GRID_DIM_X):
                        [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                    else:
                        raise ErrorDescription(6)
                except ErrorDescription as e:
                    print(e)
                    sys.exit()
            distance_backproj_arb = mod.get_function("distance_backproj_arb")
            dest = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
            x_pixel_gpu = pycuda.gpuarray.to_gpu(Xpixel.astype(np.float32))
            y_pixel_gpu = pycuda.gpuarray.to_gpu(Ypixel.astype(np.float32))
            z_pixel_gpu = pycuda.gpuarray.to_gpu(Zpixel.astype(np.float32))
            u_plane_gpu = pycuda.gpuarray.to_gpu(ki.astype(np.float32))
            v_plane_gpu = pycuda.gpuarray.to_gpu(p.astype(np.float32))
            recon_param = np.array(
                [dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                 Detector[2], angle[0], 0.0, R]).astype(np.float32)
            recon_param_gpu = drv.mem_alloc(recon_param.nbytes)
            for i in range(nViews):
                Q = self.proj[i, :, :] * dtheta
                Q = Q.flatten().astype(np.float32)
                recon_param = np.array(
                    [dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                     Detector[2], angle[i], 0.0, R]).astype(np.float32)
                # recon_param_gpu = pycuda.gpuarray.to_gpu(recon_param)
                drv.memcpy_htod(recon_param_gpu, recon_param)
                distance_backproj_arb(dest, drv.In(Q), x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, u_plane_gpu, v_plane_gpu,
                                      recon_param_gpu, block=(blockX, blockY, blockZ), grid=(gridX, gridY))
            del u_plane_gpu, v_plane_gpu, x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, recon_param_gpu
            recon = dest.get().reshape([nz, ny, nx]).astype(np.float32)
            del dest
        else:
            for i in range(nViews):
                Q = self.proj[i, :, :]
                recon += self._distance_backproj_arb(Q, Xpixel, Ypixel, Zpixel, ki, p, angle[i], 0.0,
                                                     self.params) * dtheta
        return recon

    @staticmethod
    def _distance_backproj_arb(proj, Xpixel, Ypixel, Zpixel, Uplane, Vplane, angle1, angle2, params):
        tol_min = 1e-6
        [nu, nv] = params['NumberOfDetectorPixels']
        [du, dv] = params['DetectorPixelSize']
        [dx, dy, dz] = params['ImagePixelSpacing']
        [nx, ny, nz] = params['NumberOfImage']
        dx = -1 * dx
        dy = -1 * dy
        dv = -1 * dv
        # angle1: rotation angle between point and X-axis
        # angle2: rotation angle between point and XY-palne
        Source = np.array(params['SourceInit'])
        Detector = np.array(params['DetectorInit'])
        R = sqrt(np.sum((np.array(Source) - np.array(params['PhantomCenter'])) ** 2.0))
        recon_pixelsX = Xpixel
        recon_pixelsY = Ypixel
        recon_pixelsZ = Zpixel
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        f_angle = lambda x, y: atan(x / y) if y != 0 else atan(0) if x == 0 else -pi / 2 if x < 0 else pi / 2
        fx = lambda x, y, z: x * cos(angle2) * cos(angle1) + y * cos(angle2) * sin(angle1) - z * sin(angle2) * cos(
            angle1) * sin(f_angle(x, y)) - z * sin(angle2) * sin(angle1) * cos(f_angle(x, y))
        fy = lambda x, y, z: y * cos(angle2) * cos(angle1) - x * cos(angle2) * sin(angle1) - z * sin(angle2) * cos(
            angle1) * cos(f_angle(x, y)) + z * sin(angle2) * sin(angle1) * sin(f_angle(x, y))
        fz = lambda x, y, z: z * cos(angle2) + sqrt(x ** 2 + y ** 2) * sin(angle2)
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    #                     l = sqrt(recon_pixelsX[k] ** 2 + recon_pixelsY[j] ** 2 + recon_pixelsZ[i] ** 2)
                    xc = fx(recon_pixelsX[k], recon_pixelsY[j], recon_pixelsZ[i])
                    yc = fy(recon_pixelsX[k], recon_pixelsY[j], recon_pixelsZ[i])
                    zc = fz(recon_pixelsX[k], recon_pixelsY[j], recon_pixelsZ[i])
                    #                     yc = -(recon_pixelsX[k]) * sin(angle) + (recon_pixelsY[j]) * cos(angle)
                    x1 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    y1 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    z1 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)

                    x2 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    y2 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    z2 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)

                    x3 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    y3 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    z3 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)

                    x4 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    y4 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    z4 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)

                    x5 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    y5 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    z5 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)

                    x6 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    y6 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    z6 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)

                    x7 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    y7 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    z7 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)

                    x8 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    y8 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    z8 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)

                    slope_u1 = (Source[0] - x1) / (Source[1] - y1)
                    slope_u2 = (Source[0] - x2) / (Source[1] - y2)
                    slope_u3 = (Source[0] - x3) / (Source[1] - y3)
                    slope_u4 = (Source[0] - x4) / (Source[1] - y4)
                    slope_u5 = (Source[0] - x5) / (Source[1] - y5)
                    slope_u6 = (Source[0] - x6) / (Source[1] - y6)
                    slope_u7 = (Source[0] - x7) / (Source[1] - y7)
                    slope_u8 = (Source[0] - x8) / (Source[1] - y8)
                    slopes_u = [slope_u1, slope_u2, slope_u3, slope_u4, slope_u5, slope_u6, slope_u7, slope_u8]
                    slope_l = min(slopes_u)
                    slope_r = max(slopes_u)
                    coord_u1 = (slope_l * Detector[1]) + (Source[0] - slope_r * Source[1])
                    coord_u2 = (slope_r * Detector[1]) + (Source[0] - slope_r * Source[1])
                    u_l = floor((coord_u1 - Uplane[0]) / du)
                    u_r = floor((coord_u2 - Uplane[0]) / du)
                    s_index_u = int(min(u_l, u_r))
                    e_index_u = int(max(u_l, u_r))

                    slope_v1 = (Source[2] - z1) / (Source[1] - y1)
                    slope_v2 = (Source[2] - z2) / (Source[1] - y2)
                    slope_v3 = (Source[2] - z3) / (Source[1] - y3)
                    slope_v4 = (Source[2] - z4) / (Source[1] - y4)
                    slope_v5 = (Source[2] - z5) / (Source[1] - y5)
                    slope_v6 = (Source[2] - z6) / (Source[1] - y6)
                    slope_v7 = (Source[2] - z7) / (Source[1] - y7)
                    slope_v8 = (Source[2] - z8) / (Source[1] - y8)
                    slopes_v = [slope_v1, slope_v2, slope_v3, slope_v4, slope_v5, slope_v6, slope_v7, slope_v8]
                    slope_t = min(slopes_v)
                    slope_b = max(slopes_v)
                    coord_v1 = (slope_t * Detector[2]) + (Source[2] - slope_t * Source[1])
                    coord_v2 = (slope_b * Detector[2]) + (Source[2] - slope_b * Source[1])
                    v_l = floor((coord_v1 - Vplane[0]) / dv)
                    v_r = floor((coord_v2 - Vplane[0]) / dv)
                    s_index_v = int(min(v_l, v_r))
                    e_index_v = int(min(v_l, v_r))
                    for l in range(s_index_v, e_index_v + 1):
                        if (l < 0 or l > nu):
                            continue
                        if (s_index_v == e_index_v):
                            weight1 = 1.0
                        elif (l == s_index_v):
                            weight1 = (max(coord_v1, coord_v2) - Vplane[l + 1]) / abs(coord_v1 - coord_v2)
                        elif (l == e_index_v):
                            weight1 = (Vplane[l] - min(coord_v1, coord_v2)) / abs(coord_v1 - coord_v2)
                        else:
                            weight1 = abs(dv) / abs(coord_v1 - coord_v2)
                        for m in range(s_index_u, e_index_u + 1):
                            if (m < 0 or m > nv):
                                continue
                            if (s_index_u == e_index_u):
                                weight2 = 1.0
                            elif (m == s_index_u):
                                weight2 = (Uplane[k + 1] - min(coord_u1, coord_u2)) / abs(coord_u1 - coord_u2)
                            elif (m == e_index_u):
                                weight2 = (max(coord_u1, coord_u2) - Uplane[k]) / abs(coord_u1 - coord_u2)
                            else:
                                weight2 = abs(du) / abs(coord_u1 - coord_u2)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2 * (R ** 2) / (R - yc) ** 2
        return recon

    @staticmethod
    def _distance_backproj_about_z(proj, Xpixel, Ypixel, Zpixel, Uplane, Vplane, angle, params):
        tol_min = 1e-6
        [nu, nv] = params['NumberOfDetectorPixels']
        [du, dv] = params['DetectorPixelSize']
        [dx, dy, dz] = params['ImagePixelSpacing']
        [nx, ny, nz] = params['NumberOfImage']
        dx = -1 * dx
        dy = -1 * dy
        dv = -1 * dv
        #         SAD = params['SAD']
        #         SDD = parasm['SDD']
        Source = np.array(params['SourceInit'])
        Detector = np.array(params['DetectorInit'])
        R = sqrt(np.sum((np.array(Source) - np.array(params['PhantomCenter'])) ** 2))
        recon_pixelsX = Xpixel
        recon_pixelsY = Ypixel
        recon_pixelsZ = Zpixel
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        #         recon_pixelsX = Xplane[0:-1] + dx / 2
        #         recon_pixelsY = Yplane[0:-1] + dy / 2
        #         recon_pixelsZ = Zplane[0:-1] + dz / 2
        #         [reconY, reconX] = np.meshgrid(recon_pixelsY, recon_pixlesZ)
        #         reconX_c1 = (reconX + dx / 2) * cos(angle) + (reconY + dy / 2) * sin(angle)
        #         reconX_c2 = (reconX - dx / 2) * cos(angle) + (reconY - dy / 2) * sin(angle)
        #         reconX_c3 = (reconX + dx / 2) * cos(angle) + (reconY - dy / 2) * sin(angle)
        #         reconX_c4 = (reconX - dx / 2) * cos(angle) + (reconY + dy / 2) * sin(angle)
        #
        #         reconY_c1 = -(reconX + dx / 2) * sin(angle) + (reconY + dy / 2) * cos(angle)
        #         reconY_c2 = -(reconX - dx / 2) * sin(angle) + (reconY - dy / 2) * cos(angle)
        #         reconY_c3 = -(reconX + dx / 2) * sin(angle) + (reconY - dy / 2) * cos(angle)
        #         reconY_c4 = -(reconX - dx / 2) * sin(angle) + (reconY + dy / 2) * cos(angle)
        #
        #         SlopeU_c1 = (Source[0] - reconX_c1) / (Source[1] - reconY_c1)
        #         SlopeU_c2 = (Source[0] - reconX_c2) / (Source[1] - reconY_c2)
        #         SlopeU_c3 = (Source[0] - reconX_c3) / (Source[1] - reconY_c3)
        #         SlopeU_c4 = (Source[0] - reconX_c4) / (Source[1] - reconY_c4)
        #         [reconZ, reconY] = np.meshgrid
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    yc = -(recon_pixelsX[k]) * sin(angle) + (recon_pixelsY[j]) * cos(angle)
                    x1 = (recon_pixelsX[k] + dx / 2) * cos(angle) + (recon_pixelsY[j] + dy / 2) * sin(angle)
                    y1 = -(recon_pixelsX[k] + dx / 2) * sin(angle) + (recon_pixelsY[j] + dy / 2) * cos(angle)
                    slope1 = (Source[0] - x1) / (Source[1] - y1)
                    x2 = (recon_pixelsX[k] - dx / 2) * cos(angle) + (recon_pixelsY[j] - dy / 2) * sin(angle)
                    y2 = -(recon_pixelsX[k] - dx / 2) * sin(angle) + (recon_pixelsY[j] - dy / 2) * cos(angle)
                    slope2 = (Source[0] - x2) / (Source[1] - y2)
                    x3 = (recon_pixelsX[k] + dx / 2) * cos(angle) + (recon_pixelsY[j] - dy / 2) * sin(angle)
                    y3 = -(recon_pixelsX[k] + dx / 2) * sin(angle) + (recon_pixelsY[j] - dy / 2) * cos(angle)
                    slope3 = (Source[0] - x3) / (Source[1] - y3)
                    x4 = (recon_pixelsX[k] - dx / 2) * cos(angle) + (recon_pixelsY[j] + dy / 2) * sin(angle)
                    y4 = -(recon_pixelsX[k] - dx / 2) * sin(angle) + (recon_pixelsY[j] + dy / 2) * cos(angle)
                    slope4 = (Source[0] - x4) / (Source[1] - y4)
                    slopes_u = [slope1, slope2, slope3, slope4]
                    slope_l = min(slopes_u)
                    slope_r = max(slopes_u)
                    coord_u1 = (slope_l * Detector[1]) + (Source[0] - slope_r * Source[1])
                    coord_u2 = (slope_r * Detector[1]) + (Source[0] - slope_r * Source[1])
                    u_l = floor((coord_u1 - Uplane[0]) / du)
                    u_r = floor((coord_u2 - Uplane[0]) / du)
                    s_index_u = int(min(u_l, u_r))
                    e_index_u = int(max(u_l, u_r))

                    z1 = recon_pixelsZ[i] - dz / 2
                    z2 = recon_pixelsZ[i] + dz / 2
                    slopes_v = [(Source[2] - z1) / (Source[1] - yc), (Source[2] - z2) / (Source[1] - yc)]
                    slope_t = min(slopes_v)
                    slope_b = max(slopes_v)
                    coord_v1 = (slope_t * Detector[2]) + (Source[2] - slope_t * Source[1])
                    coord_v2 = (slope_b * Detector[2]) + (Source[2] - slope_b * Source[1])
                    v_l = floor((coord_v1 - Vplane[0]) / dv)
                    v_r = floor((coord_v2 - Vplane[0]) / dv)
                    s_index_v = int(min(v_l, v_r))
                    e_index_v = int(min(v_l, v_r))
                    for l in range(s_index_v, e_index_v + 1):
                        if (s_index_v == e_index_v):
                            weight1 = 1.0
                        elif (l == s_index_v):
                            weight1 = (max(coord_v1, coord_v2) - Vplane[l + 1]) / abs(coord_v1 - coord_v2)
                        elif (l == e_index_v):
                            weight1 = (Vplane[l] - min(coord_v1, coord_v2)) / abs(coord_v1 - coord_v2)
                        else:
                            weight1 = abs(dv) / abs(coord_v1 - coord_v2)
                        for m in range(s_index_u, e_index_u + 1):
                            if (s_index_u == e_index_u):
                                weight2 = 1.0
                            elif (m == s_index_u):
                                weight2 = (Uplane[k + 1] - min(coord_u1, coord_u2)) / abs(coord_u1 - coord_u2)
                            elif (m == e_index_u):
                                weight2 = (max(coord_u1, coord_u2) - Uplane[k]) / abs(coord_u1 - coord_u2)
                            else:
                                weight2 = abs(du) / abs(coord_u1 - coord_u2)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2 * (R ** 2) / (R - yc) ** 2
        return recon

    def forward(self):
        start_time = time.time()
        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        dy = -1 * dy
        dz = -1 * dz
        Source_Init = np.array(self.params['SourceInit'])
        Detector_Init = np.array(self.params['DetectorInit'])
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = np.array(self.params['Origin'])
        PhantomCenter = np.array(self.params['PhantomCenter'])
        HelicalPitch = self.params['Pitch']
        gpu = self.params['GPU']
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2.0))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2.0))
        if (HelicalPitch > 0):
            # P = HelicalPitch * (nv * dv * SAD) / SDD
            P = HelicalPitch * (nv * dv)
            nViews = (EndAngle - StartAngle) / (2 * pi) * nViews
            log.debug(nViews)
            assert (nViews % 1.0 == 0)
            nViews = int(nViews)
        else:
            P = 0.0
            # Calculates detector center
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = angle[0:-1]
        proj = np.zeros([nViews, nv, nu], dtype=np.float32)

        # Xplane = (PhantomCenter[0] - (nx - 1) / 2.0 + range(0, nx + 1)) * dx
        # Yplane = (PhantomCenter[1] - (ny - 1) / 2.0 + range(0, ny + 1)) * dy
        # Zplane = (PhantomCenter[2] - (nz - 1) / 2.0 + range(0, nz + 1)) * dz
        Xplane = PhantomCenter[0] + (np.arange(0, nx + 1) - (nx - 1) / 2.0) * dx
        Yplane = PhantomCenter[1] + (np.arange(0, ny + 1) - (ny - 1) / 2.0) * dy
        Zplane = PhantomCenter[2] + (np.arange(0, nz + 1) - (nz - 1) / 2.0) * dz
        Xplane = Xplane - dx / 2
        Yplane = Yplane - dy / 2
        Zplane = Zplane - dz / 2
        # print(Yplane[1]-Yplane[0])
        # print(Zplane[1]-Zplane[0])
        alpha = 0
        beta = 0
        gamma = 0
        eu = [cos(gamma) * cos(alpha), sin(alpha), sin(gamma)]
        ev = [cos(gamma) * -sin(alpha), cos(gamma) * cos(alpha), sin(gamma)]
        ew = [0, 0, 1]
        # print('Variable initialization: ' + str(time.time() - start_time))

        for i in range(nViews):
            # for i in range(12, 13):
            # print(i)
            start_time = time.time()
            Source = np.array([-SAD * sin(angle[i]), SAD * cos(angle[i]),
                               Source_Init[2] + P * angle[i] / (2 * pi)])  # z-direction rotation
            Detector = np.array(
                [(SDD - SAD) * sin(angle[i]), -(SDD - SAD) * cos(angle[i]), Detector_Init[2] + P * angle[i] / (2 * pi)])
            # DetectorLength = np.array(
            #    [np.arange(floor(-nu / 2), floor(nu / 2) + 1) * du, np.arange(floor(-nv / 2), floor(nv / 2) + 1) * dv])
            # DetectorVectors = [eu, ev, ew]
            if (self.params['DetectorShape'] == 'Flat'):
                [DetectorIndex, DetectorBoundary] = self.FlatDetectorConstruction(Source, Detector, SDD, angle[i])
                # print(DetectorBoundary.shape)
                # print(DetectorIndex[0,128,:])
                # sys.exit()
            elif (self.params['DetectorShape'] == 'Curved'):
                [DetectorIndex, DetectorBoundary] = self.CurvedDetectorConstruction(Source, Detector, SDD, angle[i])
            else:
                print('Detector shape is not supproted!')
                sys.exit()
            # print('Detector initialization: ' + str(time.time() - start_time))
            if (self.params['Method'] == 'Distance'):
                start_time = time.time()
                proj[i, :, :] = self.distance(DetectorIndex, DetectorBoundary, Source, Detector, angle[i], Xplane,
                                              Yplane, Zplane)
                # print('Total projection: ' + str(time.time() - start_time))
            elif (self.params['Method'] == 'Ray'):
                proj[i, :, :] = self.ray(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)
            # print('time taken: ' + str(time.time() - start_time) + '\n')
        self.proj = proj

    def distance(self, DetectorIndex, DetectorBoundary, Source, Detector, angle, Xplane, Yplane, Zplane):
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        proj = np.zeros([nv, nu], dtype=np.float32)
        if self.params['GPU']:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            distance_proj_on_y_gpu = mod.get_function("distance_project_on_y2")
            distance_proj_on_x_gpu = mod.get_function("distance_project_on_x2")
            distance_proj_on_z_gpu = mod.get_function("distance_project_on_z2")
            image = self.image.flatten().astype(np.float32)
            dest = pycuda.gpuarray.to_gpu(proj.flatten().astype(np.float32))
            x_plane_gpu = pycuda.gpuarray.to_gpu(Xplane.astype(np.float32))
            y_plane_gpu = pycuda.gpuarray.to_gpu(Yplane.astype(np.float32))
            z_plane_gpu = pycuda.gpuarray.to_gpu(Zplane.astype(np.float32))
        start_time = time.time()
        DetectorBoundaryU1 = np.array(
            [DetectorBoundary[0, 0:-1, 0:-1], DetectorBoundary[1, 0:-1, 0:-1], DetectorIndex[2, :, :]])
        DetectorBoundaryU2 = np.array(
            [DetectorBoundary[0, 1:, 1:], DetectorBoundary[1, 1:, 1:], DetectorIndex[2, :, :]])
        DetectorBoundaryV1 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorBoundary[2, 1:, 1:]])
        DetectorBoundaryV2 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorBoundary[2, 0:-1, 0:-1]])
        # DetectorBoundaryU1 = np.array(
        #     [DetectorIndex[0, :, :] - cos(angle) * du / 2, DetectorIndex[1, :, :] - sin(angle) * du / 2,
        #      DetectorIndex[2, :, :]])
        # DetectorBoundaryU2 = np.array(
        #     [DetectorIndex[0, :, :] + cos(angle) * du / 2, DetectorIndex[1, :, :] + sin(angle) * du / 2,
        #      DetectorIndex[2, :, :]])
        # DetectorBoundaryV1 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorIndex[2, :, :] - dv / 2])
        # DetectorBoundaryV2 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorIndex[2, :, :] + dv / 2])
        SDD = sqrt(np.sum((Source - Detector) ** 2.0))
        ray_angles = atan(sqrt(
            (DetectorIndex[0, :, :] - Detector[0]) ** 2.0 + (DetectorIndex[1, :, :] - Detector[1]) ** 2.0 + (
                    DetectorIndex[2, :, :] - Detector[2]) ** 2.0) / SDD)
        # ray_normalization = cos(ray_angles)
        ray_normalization = 1.0
        if (abs(dy / dx) >= abs(Source[1] - Detector[1]) / (abs(Source[0] - Detector[0]) + 1e-4) and
                abs(dz / dx) >= abs(Source[2] - Detector[2]) / (abs(Source[0] - Detector[0]) + 1e-4)):
            SlopesU1 = (Source[1] - DetectorBoundaryU1[1, :, :]) / (Source[0] - DetectorBoundaryU1[0, :, :])
            InterceptsU1 = -SlopesU1 * Source[0] + Source[1]
            SlopesU2 = (Source[1] - DetectorBoundaryU2[1, :, :]) / (Source[0] - DetectorBoundaryU2[0, :, :])
            InterceptsU2 = -SlopesU2 * Source[0] + Source[1]
            SlopesV1 = (Source[2] - DetectorBoundaryV1[2, :, :]) / (Source[0] - DetectorBoundaryV1[0, :, :])
            InterceptsV1 = -SlopesV1 * Source[0] + Source[2]
            SlopesV2 = (Source[2] - DetectorBoundaryV2[2, :, :]) / (Source[0] - DetectorBoundaryV2[0, :, :])
            InterceptsV2 = -SlopesV2 * Source[0] + Source[2]
            intersection_slope1 = (Source[1] - DetectorIndex[1, :, :]) / (Source[0] - DetectorIndex[0, :, :])
            intersection_slope2 = (Source[2] - DetectorIndex[2, :, :]) / (Source[0] - DetectorIndex[0, :, :])
            intersection_length = abs(dx) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))

            if (self.params['GPU']):
                TotalSize = nu * nv * nx
                if (TotalSize < MAX_THREAD_PER_BLOCK):
                    blockX = nu * nv * nx
                    blockY = 1
                    blockZ = 1
                    gridX = 1
                    gridY = 1
                else:
                    blockX = 32
                    blockY = 32
                    blockZ = 1
                    GridSize = ceil(TotalSize / (blockX * blockY))
                    try:
                        if (GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()
                proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                slope_y1_gpu = pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                slope_y2_gpu = pycuda.gpuarray.to_gpu(SlopesU2.flatten().astype(np.float32))
                slope_z1_gpu = pycuda.gpuarray.to_gpu(SlopesV1.flatten().astype(np.float32))
                slope_z2_gpu = pycuda.gpuarray.to_gpu(SlopesV2.flatten().astype(np.float32))
                intercept_y1_gpu = pycuda.gpuarray.to_gpu(InterceptsU1.flatten().astype(np.float32))
                intercept_y2_gpu = pycuda.gpuarray.to_gpu(InterceptsU2.flatten().astype(np.float32))
                intercept_z1_gpu = pycuda.gpuarray.to_gpu(InterceptsV1.flatten().astype(np.float32))
                intercept_z2_gpu = pycuda.gpuarray.to_gpu(InterceptsV2.flatten().astype(np.float32))
                proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                distance_proj_on_x_gpu(dest, drv.In(image), slope_y1_gpu, slope_y2_gpu, slope_z1_gpu,
                                       slope_z2_gpu, intercept_y1_gpu, intercept_y2_gpu, intercept_z1_gpu,
                                       intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del slope_y1_gpu, slope_y2_gpu, slope_z1_gpu, slope_z2_gpu, intercept_y1_gpu, intercept_y2_gpu, intercept_z1_gpu, intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * (intersection_length / ray_normalization)
                del dest
            else:
                for ix in range(nx):
                    CoordY1 = SlopesU1 * (Xplane[ix] + dx / 2) + InterceptsU1
                    CoordY2 = SlopesU2 * (Xplane[ix] + dx / 2) + InterceptsU2
                    CoordZ1 = SlopesV1 * (Xplane[ix] + dx / 2) + InterceptsV1
                    CoordZ2 = SlopesV2 * (Xplane[ix] + dx / 2) + InterceptsV2
                    image_y1 = floor((CoordY1 - Yplane[0] + 0) / dy)
                    image_y2 = floor((CoordY2 - Yplane[0] + 0) / dy)
                    image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                    image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                    proj += self._distance_project_on_x(self.image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane, Zplane,
                                                        image_y1, image_y2, image_z1, image_z2, dy, dz, ix) * (
                                    intersection_length / ray_normalization)


        elif (abs(dx / dy) >= abs(Source[0] - Detector[0]) / (abs(Source[1] - Detector[1]) + 1e-4) and
              abs(dz / dy) >= abs(Source[2] - Detector[2]) / (abs(Source[1] - Detector[1]) + 1e-4)):
            start_time = time.time()
            SlopesU1 = (Source[0] - DetectorBoundaryU1[0, :, :]) / (Source[1] - DetectorBoundaryU1[1, :, :])
            InterceptsU1 = -SlopesU1 * Source[1] + Source[0]
            SlopesU2 = (Source[0] - DetectorBoundaryU2[0, :, :]) / (Source[1] - DetectorBoundaryU2[1, :, :])
            InterceptsU2 = -SlopesU2 * Source[1] + Source[0]
            SlopesV1 = (Source[2] - DetectorBoundaryV1[2, :, :]) / (Source[1] - DetectorBoundaryV1[1, :, :])
            InterceptsV1 = -SlopesV1 * Source[1] + Source[2]
            SlopesV2 = (Source[2] - DetectorBoundaryV2[2, :, :]) / (Source[1] - DetectorBoundaryV2[1, :, :])
            InterceptsV2 = -SlopesV2 * Source[1] + Source[2]
            # print('Calculate line: ' + str(time.time() - start_time))
            intersection_slope1 = (Source[0] - DetectorIndex[0, :, :]) / (Source[1] - DetectorIndex[1, :, :])
            intersection_slope2 = (Source[2] - DetectorIndex[2, :, :]) / (Source[1] - DetectorIndex[1, :, :])
            intersection_length = abs(dy) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))
            if (self.params['GPU']):
                TotalSize = nu * nv * ny
                if (TotalSize < MAX_THREAD_PER_BLOCK):
                    blockX = nu * nv * ny
                    blockY = 1
                    blockZ = 1
                    gridX = 1
                    gridY = 1
                else:
                    blockX = 32
                    blockY = 32
                    blockZ = 1
                    GridSize = ceil(TotalSize / (blockX * blockY))
                    try:
                        if (GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()
                proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                slope_x1_gpu = pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                slope_x2_gpu = pycuda.gpuarray.to_gpu(SlopesU2.flatten().astype(np.float32))
                slope_z1_gpu = pycuda.gpuarray.to_gpu(SlopesV1.flatten().astype(np.float32))
                slope_z2_gpu = pycuda.gpuarray.to_gpu(SlopesV2.flatten().astype(np.float32))
                intercept_x1_gpu = pycuda.gpuarray.to_gpu(InterceptsU1.flatten().astype(np.float32))
                intercept_x2_gpu = pycuda.gpuarray.to_gpu(InterceptsU2.flatten().astype(np.float32))
                intercept_z1_gpu = pycuda.gpuarray.to_gpu(InterceptsV1.flatten().astype(np.float32))
                intercept_z2_gpu = pycuda.gpuarray.to_gpu(InterceptsV2.flatten().astype(np.float32))
                proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                distance_proj_on_y_gpu(dest, drv.In(image), slope_x1_gpu, slope_x2_gpu, slope_z1_gpu,
                                       slope_z2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_z1_gpu,
                                       intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del slope_x1_gpu, slope_x2_gpu, slope_z1_gpu, slope_z2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_z1_gpu, intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * (intersection_length / ray_normalization)
                del dest
            else:
                for iy in range(ny):
                    start_time = time.time()
                    CoordX1 = SlopesU1 * (Yplane[iy] + dy / 2) + InterceptsU1
                    CoordX2 = SlopesU2 * (Yplane[iy] + dy / 2) + InterceptsU2
                    CoordZ1 = SlopesV1 * (Yplane[iy] + dy / 2) + InterceptsV1
                    CoordZ2 = SlopesV2 * (Yplane[iy] + dy / 2) + InterceptsV2
                    image_x1 = floor((CoordX1 - Xplane[0] + 0) / dx)
                    image_x2 = floor((CoordX2 - Xplane[0] + 0) / dx)
                    image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                    image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                    proj += self._distance_project_on_y(self.image, CoordX1, CoordX2, CoordZ1, CoordZ2, Xplane, Zplane,
                                                        image_x1, image_x2, image_z1, image_z2, dx, dz, iy) * (
                                    intersection_length / ray_normalization)

        elif (abs(dx / dz) >= abs(Source[0] - Detector[0]) / (abs(Source[2] - Detector[2]) + 1e-4) and
              abs(dy / dz) >= abs(Source[1] - Detector[1]) / (abs(Source[2] - Detector[2]) + 1e-4)):
            SlopesU1 = (Source[0] - DetectorBoundaryU1[0, :, :]) / (Source[2] - DetectorBoundaryU1[2, :, :])
            InterceptsU1 = -SlopesU1 * Source[2] + Source[0]
            SlopesU2 = (Source[0] - DetectorBoundaryU2[0, :, :]) / (Source[2] - DetectorBoundaryU2[2, :, :])
            InterceptsU2 = -SlopesU2 * Source[2] + Source[0]
            SlopesV1 = (Source[1] - DetectorBoundaryV1[1, :, :]) / (Source[2] - DetectorBoundaryV1[2, :, :])
            InterceptsV1 = -SlopesV1 * Source[2] + Source[1]
            SlopesV2 = (Source[1] - DetectorBoundaryV2[1, :, :]) / (Source[2] - DetectorBoundaryV2[2, :, :])
            InterceptsV2 = -SlopesV2 * Source[2] + Source[1]
            intersection_slope1 = (Source[0] - DetectorIndex[0, :, :]) / (Source[2] - DetectorIndex[2, :, :])
            intersection_slope2 = (Source[1] - DetectorIndex[1, :, :]) / (Source[2] - DetectorIndex[2, :, :])
            intersection_length = abs(dz) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))
            if (self.params['GPU']):
                TotalSize = nu * nv * nz
                if (TotalSize < MAX_THREAD_PER_BLOCK):
                    blockX = nu * nv * nz
                    blockY = 1
                    blockZ = 1
                    gridX = 1
                    gridY = 1
                else:
                    blockX = 32
                    blockY = 32
                    blockZ = 1
                    GridSize = ceil(TotalSize / (blockX * blockY))
                    try:
                        if (GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()
                proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                slope_x1_gpu = pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                slope_x2_gpu = pycuda.gpuarray.to_gpu(SlopesU2.flatten().astype(np.float32))
                slope_y1_gpu = pycuda.gpuarray.to_gpu(SlopesV1.flatten().astype(np.float32))
                slope_y2_gpu = pycuda.gpuarray.to_gpu(SlopesV2.flatten().astype(np.float32))
                intercept_x1_gpu = pycuda.gpuarray.to_gpu(InterceptsU1.flatten().astype(np.float32))
                intercept_x2_gpu = pycuda.gpuarray.to_gpu(InterceptsU2.flatten().astype(np.float32))
                intercept_y1_gpu = pycuda.gpuarray.to_gpu(InterceptsV1.flatten().astype(np.float32))
                intercept_y2_gpu = pycuda.gpuarray.to_gpu(InterceptsV2.flatten().astype(np.float32))
                proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                distance_proj_on_z_gpu(dest, drv.In(image), slope_x1_gpu, slope_x2_gpu, slope_y1_gpu,
                                       slope_y2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_y1_gpu,
                                       intercept_y2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del slope_x1_gpu, slope_x2_gpu, slope_y1_gpu, slope_y2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_y1_gpu, intercept_y2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * (intersection_length / ray_normalization)
                del dest
            else:
                for iz in range(nz):
                    CoordX1 = SlopesU1 * Zplane[iz] + dz / 2 + InterceptsU1
                    CoordX2 = SlopesU2 * Zplane[iz] + dz / 2 + InterceptsU2
                    CoordY1 = SlopesV1 * Zplane[iz] + dz / 2 + InterceptsV1
                    CoordY2 = SlopesV2 * Zplane[iz] + dz / 2 + InterceptsV2
                    image_x1 = floor(CoordX1 - Xplane[0] + dx) / dx
                    image_x2 = floor(CoordX2 - Xplane[0] + dx) / dx
                    image_y1 = floor(CoordY1 - Yplane[0] + dy) / dy
                    image_y2 = floor(CoordY2 - Yplane[0] + dy) / dy
                    proj += self._distance_project_on_z(self.image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane,
                                                        image_x1, image_x2, image_y1, image_y2, dx, dy, iz) * (
                                    intersection_length / ray_normalization)
        else:
            raise ErrorDescription(7)

        return proj

    @staticmethod
    def _distance_project_on_y(image, CoordX1, CoordX2, CoordZ1, CoordZ2, Xplane, Zplane, image_x1, image_x2, image_z1,
                               image_z2, dx, dz, iy):
        tol_min = 1e-6
        tol_max = 1e7
        proj = np.zeros(CoordX1.shape, dtype=np.float32)
        start_time = time.time()
        for i in range(CoordX1.shape[0]):
            for j in range(CoordX1.shape[1]):
                p_value = 0
                s_index_x = min(image_x1[i, j], image_x2[i, j])
                e_index_x = max(image_x1[i, j], image_x2[i, j])
                s_index_z = min(image_z1[i, j], image_z2[i, j])
                e_index_z = max(image_z2[i, j], image_z2[i, j])
                for k in range(int(s_index_x), int(e_index_x) + 1):
                    if (k < 0 or k > image.shape[0] - 1):
                        continue
                    if (s_index_x == e_index_x):
                        weight1 = 1
                    elif (k == s_index_x):
                        # print(k,s_index_x,e_index_x,Xplane[k+1],CoordX1[i,j],CoordX2[i,j])
                        weight1 = (Xplane[k + 1] - min(CoordX1[i, j], CoordX2[i, j])) / abs(
                            CoordX1[i, j] - CoordX2[i, j])
                    elif (k == e_index_x):
                        # print(k,s_index_x,e_index_x)
                        # print(Xplane[k],CoordX1[i,j],CoordX2[i,j])
                        weight1 = (max(CoordX1[i, j], CoordX2[i, j]) - Xplane[k]) / abs(CoordX1[i, j] - CoordX2[i, j])
                    else:
                        weight1 = abs(dx) / abs(CoordX1[i, j] - CoordX2[i, j])
                    for l in range(int(s_index_z), int(e_index_z) + 1):
                        if (l < 0 or l > image.shape[2] - 1):
                            continue
                        if (s_index_z == e_index_z):
                            weight2 = 1
                        elif (l == s_index_z):
                            # print(s_index_z,e_index_z,Zplane[l+1],CoordZ1[i,j],CoordZ2[i,j])
                            weight2 = (max(CoordZ1[i, j], CoordZ2[i, j]) - Zplane[l + 1]) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        elif (l == e_index_z):
                            # print('1')
                            weight2 = (Zplane[l] - min(CoordZ1[i, j], CoordZ2[i, j])) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        else:
                            weight2 = abs(dz) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        # print(weight1,weight2)
                        assert (weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[l][iy][k]
                proj[i, j] = p_value
        # print('Projection for a loop: ' + str(time.time() - start_time))
        return proj

    @staticmethod
    def _distance_project_on_x(image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane, Zplane, image_y1, image_y2, image_z1,
                               image_z2, dy, dz, ix):
        tol_min = 1e-6
        tol_max = 1e7
        proj = np.zeros(CoordY1.shape, dtype=np.float32)
        for i in range(CoordY1.shape[0]):
            for j in range(CoordY1.shape[1]):
                p_value = 0
                s_index_y = min(image_y1[i, j], image_y2[i, j])
                e_index_y = max(image_y1[i, j], image_y2[i, j])
                s_index_z = min(image_z1[i, j], image_z2[i, j])
                e_index_z = max(image_z1[i, j], image_z2[i, j])
                for k in range(int(s_index_y), int(e_index_y) + 1):
                    if (k < 0 or k > image.shape[1] - 1):
                        continue
                    if (s_index_y == e_index_y):
                        weight1 = 1
                    elif (k == s_index_y):
                        weight1 = (max(CoordY1[i, j], CoordY2[i, j]) - Yplane[k + 1]) / abs(
                            CoordY1[i, j] - CoordY2[i, j])
                    elif (k == e_index_y):
                        weight1 = (Yplane[k] - min(CoordY1[i, j], CoordY2[i, j])) / abs(CoordY1[i, j] - CoordY2[i, j])
                    else:
                        weight1 = abs(dy) / abs(CoordY1[i, j] - CoordY2[i, j])
                    # if(abs(weight1) - 0 < tol_min):
                    #    weight1 = 0
                    for l in range(int(s_index_z), int(e_index_z) + 1):
                        if (l < 0 or l > image.shape[2] - 1):
                            continue
                        if (s_index_z == e_index_z):
                            weight2 = 1
                        elif (l == s_index_z):
                            weight2 = (max(CoordZ1[i, j], CoordZ2[i, j]) - Zplane[l + 1]) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        elif (l == e_index_z):
                            weight2 = (Zplane[l] - min(CoordZ1[i, j], CoordZ2[i, j])) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        else:
                            weight2 = abs(dz) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        # print(s_index_z,e_index_z,Zplane[l+1],Zplane[l],CoordZ1[i,j],CoordZ2[i,j])
                        # if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        # print(weight1,weight2)
                        assert (weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[l][k][ix]
                proj[i, j] = p_value
        return proj

    @staticmethod
    def _distance_project_on_z(image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane, image_x1, image_X2, image_y1,
                               image_y2, dx, dy, iz):
        tol_min = 1e-6
        tol_max = 1e7
        proj = np.zeros(CoordX1.shape, dtype=np.float32)
        for i in range(CoordX1.shape[0]):
            for j in range(CoordX1.shape[1]):
                p_value = 0
                s_index_x = min(image_x1[i, j], image_x2[i, j])
                e_index_x = max(image_x1[i, j], image_x2[i, j])
                s_index_y = min(image_y1[i, j], image_y2[i, j])
                e_index_y = max(image_y1[i, j], image_y2[i, j])
                for k in range(int(s_index_x), int(e_index_x) + 1):
                    if (k < 0 or k > image.shape[0] - 1):
                        continue
                    if (s_index_x == e_index_x):
                        weight1 = 1
                    elif (k == s_index_x):
                        weight1 = (Xplane[k + 1] - max(CoordX1[i, j], CoordX2[i, j])) / abs(
                            CoordX1[i, j] - CoordX2[i, j])
                    elif (k == e_index_x):
                        weight1 = (min(CoordY1[i, j], CoordY2[i, j]) - Xplane[k]) / abs(CoordX1[i, j] - CoordX2[i, j])
                    else:
                        weight1 = abs(dx) / abs(CoordX1[i, j] - CoordX2[i, j])
                    # if(abs(weight1) - 0 < tol_min):
                    #    weight1 = 0
                    for l in range(int(s_index_y), int(e_index_y) + 1):
                        if (l < 0 or l > image.shape[1] - 1):
                            continue
                        if (s_index_z == e_index_z):
                            weight2 = 1
                        elif (l == s_index_y):
                            weight2 = (max(CoordY1[i, j], CoordY2[i, j]) - Yplane[l + 1]) / abs(
                                CoordY1[i, j] - CoordY2[i, j])
                        elif (l == e_index_y):
                            weight2 = (Yplane[l] - min(CoordY1[i, j], CoordY2[i, j])) / abs(
                                CoordY1[i, j] - CoordY2[i, j])
                        else:
                            weight2 = abs(dy) / abs(CoordY1[i, j] - CoordY2[i, j])
                        # print(s_index_z,e_index_z,Zplane[l+1],Zplane[l],CoordZ1[i,j],CoordZ2[i,j])
                        # if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        # print(weight1,weight2)
                        assert (weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[iz][l][k]
                proj[i, j] = p_value
        return proj

    def ray(self):
        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [dv, du] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        Source_Init = np.array(self.params['SourceInit'])
        Detector_Init = np.array(self.params['DetectorInit'])
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = np.array(self.params['Origin'])
        PhantomCenter = np.array(self.params['PhantomCenter'])
        gpu = self.params['GPU']

        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2))
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = theta[0:-1]
        Xplane = PhantomCenter[0] + (range(0, nx) - (nx - 1) / 2.0) * dx
        Yplane = PhantomCenter[1] + (range(0, ny) - (ny - 1) / 2.0) * dy
        Zplane = PhantomCenter[2] + (range(0, nz) - (nz - 1) / 2.0) * dz
        Xplane = Xplane - dx / 2
        Yplane = Yplane - dy / 2
        Zplane = Zplane - dz / 2
        proj = np.zeros([nViews, nu, nv], dtype=np.float32)
        for angle in theta:
            # starting from x-axis and rotating ccw
            SourceX = -SAD * sin(angle)
            SourceY = SAD * cos(angle)
            SourceZ = 0
            DetectorX = (SDD - SAD) * sin(angle)
            DetectorY = -(SDD - SAD) * cos(angle)
            DetectorZ = 0
            DetectorLengthU = range(floor(-nu / 2), floor(nu / 2)) * du
            DetectorLengthV = range(floor(-nv / 2), floor(nv / 2)) * dv
            if (abs(tan(angle)) < tol_min):
                DetectorIndex = [DetectorX + DetectlrLengthU]
                DetectorIndexZ = DetectorZ - DetectorLengthV
            elif (tan(angle) >= tol_max):
                DetectorIndex = [DetectorY + DetectorLengthU]
                DetectorIndexZ = DetectorZ - DetectorLengthV
            else:
                xx = sqrt(DetectorLengthU ** 2 / (1 + tan(angle) ** 2))
                yy = tan(angle) * sqrt(DetectorLengthU ** 2 / (1 + tan(angle) ** 2))
                DetectorIndex = [DetectorX * np.sign(DetectorLengthU * xx), ]
            if (DetectorY > 0):
                DetectorIndex = DetectoIndex[:, ]
            DetectorIndex = DetectorIndex[:, 1:-2]
            DetectorIndexZ = DetectorIndexZ[1:-2]
            if (gpu):
                pass
            else:
                pass

        if (save):
            proj.tofile(write_filename, sep='', format='')

        return proj
