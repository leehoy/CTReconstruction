import os
import numpy as np
from scipy.interpolate import interp2d, griddata, RegularGridInterpolator
import glob
import matplotlib.pyplot as plt
import time
# import pycuda.driver as drv
# import pycuda.autoinit
# from pycuda.compiler import SourceModule

# function alias starts
sin = np.sin
cos = np.cos
atan = np.arctan
fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
sinc = np.sinc
sqrt = np.sqrt
real = np.real
ceil = np.ceil
log2 = np.log2
pi = np.pi
# function alias ends


class ErrorDescription:
    def __init__(self, value):
        if(value == 1):
            self.msg = 'Unknown variables'
        elif(value == 2):
            self.msg = 'Unknown data precision'
        elif(value == 3):
            self.msg = 'Number of file is different from number of projection data required'
        elif(value == 4):
            self.msg = 'Cutoff have to be pose between 0 and 0.5'
        elif(value == 5):
            self.msg = 'Smooth have to be pose between 0 and 1'
        else:
            self.msg = 'Unknown error'

    def __str__(self):
        return self.msg


class ConeBeam:
    def __init__(self, filename):
        self.params = {'SourceToDetector': 0, 'SourceToAxis': 0, 'DataPath': '',
                       'precision': '', 'AngleCoverage': 0, 'ReconX': 0, 'ReconY': 0,
                       'ReconZ': 0, 'DetectorPixelHeight': 0, 'DetectorPixelWidth': 0,
                       'DetectorWidth': 0, 'DetectorHeight': 0, 'NumberOfViews': 0,
                       'fov': 0, 'fovz': 0}
        f = open(filename)
        try:
            while True:
                line = f.readline()
                if not line:
                    break
                p = line.split(':')[0].strip()
                if(p in self.params.keys() or p == 'ReconVolume'):
                    value = line.split(':')[1].strip()
                    if(p == 'AngleCoverage'):
                        self.params[p] = float(value) * np.pi / 180.0
                    elif(p == 'ReconVolume'):
                        value = value.split('*')
                        self.params['ReconX'] = int(value[0])
                        self.params['ReconY'] = int(value[1])
                        self.params['ReconZ'] = int(value[2])
                    elif(p == 'DataPath' or p == 'precision'):
                        # print value
                        self.params[p] = value
                    else:
                        # print value
                        self.params[p] = float(value)
                else:
                    raise ErrorDescription(1)
        except ErrorDescription as e:
            print(e)
        finally:
            f.close()
        self.proj = np.zeros([int(self.params['DetectorHeight']),
                              int(self.params['DetectorWidth']), int(self.params['NumberOfViews'])])
        self.recon = np.zeros(
            [self.params['ReconX'], self.params['ReconY'], self.params['ReconZ']])

    def Reconstruction(self, savefile):
        ''' TO DOs: zero pad filter and projection data'''
        R = self.params['SourceToAxis']
        D = self.params['SourceToDetector'] - R
        nx = int(self.params['DetectorWidth'])
        ny = int(self.params['DetectorHeight'])
        ns = int(self.params['NumberOfViews'])
        DetectorPixelWidth = self.params['DetectorPixelWidth']
        DetectorPixelHeight = self.params['DetectorPixelHeight']
        recon = np.zeros(
            [self.params['ReconX'], self.params['ReconY'], self.params['ReconZ']])
        DetectorSize = [nx * DetectorPixelWidth, ny * DetectorPixelHeight]
        ZeroPaddedLength = int(2 ** (ceil(log2(2 * (nx - 1)))))
        fov = 2.0 * R * sin(atan(DetectorSize[0] / 2.0 / (D + R)))
        fovz = 2.0 * R * sin(atan(DetectorSize[1] / 2.0 / (D + R)))
        self.params['fov'] = fov
        self.params['fovz'] = fovz
        x = np.linspace(-fov / 2.0, fov / 2.0, self.params['ReconX'])
        y = np.linspace(-fov / 2.0, fov / 2.0, self.params['ReconY'])
        z = np.linspace(-fovz / 2.0, fovz / 2.0, self.params['ReconZ'])
        [xx, yy] = np.meshgrid(x, y)
        ReconZ = self.params['ReconZ']
        ProjectionAngle = np.linspace(0, self.params['AngleCoverage'], ns + 1)
        ProjectionAngle = ProjectionAngle[0:-1]
        dtheta = ProjectionAngle[1] - ProjectionAngle[0]
        assert(len(ProjectionAngle == ns))
        print('Reconstruction starts')
        # ki = np.arange(0 - (nx - 1) / 2, nx - (nx - 1) / 2)
        # p = np.arange(0 - (ny - 1) / 2, ny - (ny - 1) / 2)
        ki = np.arange(0, nx) - (nx - 1) / 2.0
        p = np.arange(0, ny) - (ny - 1) / 2.0
        ki = ki * DetectorPixelWidth
        p = p * DetectorPixelHeight
        cutoff = 0.3
        FilterType = 'hamming'
        filter = ConeBeam.Filter(
            ZeroPaddedLength + 1, DetectorPixelWidth * R / (D + R), FilterType, cutoff)
        ki = (ki * R) / (R + D)
        p = (p * R) / (R + D)
        [kk, pp] = np.meshgrid(ki, p)
# 		sample_points = np.vstack((pp.flatten(), kk.flatten())).T
        weight = R / (sqrt(R ** 2 + kk ** 2 + pp ** 2))
        for i in range(0, ns):
            angle = ProjectionAngle[i]
            if i == 0:
                print("1st projection")
            elif i == 1:
                print("2nd projection")
            elif i == 2:
                print("3rd projection")
            else:
                print(i, 'th projection')
            WeightedProjection = weight * self.proj[:, :, i]
            Q = np.zeros(WeightedProjection.shape)
            for k in range(ny):
                tmp = real(ifft(
                    ifftshift(filter * fftshift(fft(WeightedProjection[k, :], ZeroPaddedLength)))))
                Q[k, :] = tmp[0:nx]
            InterpolationFunction = RegularGridInterpolator(
                (p, ki), Q, bounds_error=False, fill_value=0)
            t = xx * cos(angle) + yy * sin(angle)
            s = -xx * sin(angle) + yy * cos(angle)
#  			for l in range(0, ReconZ):
            for l in range(255, 256):
                InterpX = (R * t) / (R - s)
                InterpY = (R * z[l]) / (R - s)
                InterpW = (R ** 2) / ((R - s) ** 2)
                pts = np.vstack((InterpY.flatten(), InterpX.flatten())).T
                vq = InterpolationFunction(pts)
                recon[l, :, :] += InterpW * dtheta * \
                    vq.reshape([self.params['ReconX'], self.params['ReconY']])
# 				Interpolgpu(drv.Out(dest),drv.In(Q),block=())
            # Interpolation required

        self.recon = recon.astype(np.float32)
        recon.tofile(savefile, sep='', format='')
        # f = open('condition.txt')
        # f.close()

    def LoadData(self):
        ns = int(self.params['NumberOfViews'])
        nx = int(self.params['DetectorWidth'])
        ny = int(self.params['DetectorHeight'])
        path = self.params['DataPath']

        filelist = sorted(glob.glob(path + '*.dat'))
        try:
            if(self.params['precision'] == 'float32'):
                precision = np.float32
            elif(self.params['precision'] == 'float64'):
                precision = np.float64
            elif(self.params['precision'] == 'int32'):
                precision = np.int32
            elif(self.params['precision'] == 'int64'):
                precision = np.int64
            else:
                raise ErrorDescription(2)
            if(not len(filelist) == ns):
                raise ErrorDescription(3)
            else:
                c = 0
                for f in filelist:
                    image = np.fromfile(f, dtype=precision).reshape([ny, nx])
                    self.proj[:, :, c] = image
                    c += 1
        except ErrorDescription as e:
            print(e)

    def Forward(self):
        pass

    @staticmethod
    def Filter(N, pixel_size, FilterType, cutoff):
        '''
        TO DO: Ram-Lak filter implementation
                   Argument for name of filter
        '''
        try:
            if cutoff > 0.5 or cutoff < 0:
                raise ErrorDescription(4)
        except ErrorDescription as e:
            print(e)
        x = np.arange(0, N) - (N - 1) / 2
        h = np.zeros(len(x))
        h[np.where(x == 0)] = 1 / (8 * pixel_size ** 2)
        odds = np.where(x % 2 == 1)
        h[odds] = -0.5 / (pi * pixel_size * x[odds]) ** 2
        h = h[0:-1]
        filter = abs(fftshift(fft(h))) * 2
        w = 2 * pi * x[0:-1] / (N - 1)
        print(filter.shape, w.shape)
        if FilterType == 'ram-lak':
            pass  # Do nothing
        elif FilterType == 'shepp-logan':
            zero = np.where(w == 0)
            tmp = filter[zero]
            filter = filter * sin(w / (2 * cutoff)) / (w / (2 * cutoff))
            filter[zero] = tmp * sin(w[zero] / (2 * cutoff))
        elif FilterType == 'cosine':
            filter = filter * cos(w / (2 * cutoff))
        elif FilterType == 'hamming':
            filter = filter * (0.54 + 0.46 * (cos(w / cutoff)))
        elif FilterType == 'hann':
            filter = filter * (0.5 + 0.5 * cos(w / cutoff))

        filter[np.where(abs(w) > pi * cutoff)] = 0
        return filter


def main():
    start_time = time.time()
    filename = './ReconstructionParams.txt'
    R = ConeBeam(filename)
    R.LoadData()
    R.Reconstruction('./Recon.dat')
    print('%s seconds taken\n' % (time.time() - start_time))
    print(R.recon.shape)
    plt.imshow(R.recon[255, :, :], cmap='gray',
               vmin=R.recon.min(), vmax=R.recon.max())
    plt.show()


if __name__ == '__main__':
    main()
