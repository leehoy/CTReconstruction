import os
import numpy as np
from scipy.interpolate import interp1d, griddata
import glob
import matplotlib.pyplot as plt
import time

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
            self.msg = 'Cutoff value has to be pose between 0 and 0.5'
        elif(value == 5):
            self.msg = 'Smooth value has to be pose between 0 and 1'
        else:
            self.msg = 'Unknown error'

    def __str__(self):
        return self.msg


class FanBeam:
    def __init__(self, filename):
        self.params = {'SourceToDetector': 0, 'SourceToAxis': 0, 'DataPath': '',
                       'precision': '', 'AngleCoverage': 0, 'ReconX': 0, 'ReconY': 0,
                       'DetectorPixelWidth': 0, 'DetectorWidth': 0, 'DetectorHeight': 0,
                       'NumberOfViews': 0, 'fov': 0, 'fovz': 0}
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
                        self.params[p] = float(value) * pi / 180.0
                    elif(p == 'ReconVolume'):
                        value = value.split('*')
                        self.params['ReconX'] = int(value[0])
                        self.params['ReconY'] = int(value[1])
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
        self.sino = np.zeros([int(self.params['NumberOfViews']), int(self.params['DetectorWidth']),
                              int(self.params['DetectorHeight'])])
        self.recon = np.zeros([self.params['ReconX'], self.params['ReconY'], int(
            self.params['DetectorHeight'])])

    def Reconstruction(self, savefile):
        R = self.params['SourceToAxis']
        D = self.params['SourceToDetector'] - R
        nx = int(self.params['DetectorWidth'])
        ny = int(self.params['DetectorHeight'])
        ns = int(self.params['NumberOfViews'])
        DetectorPixelWidth = self.params['DetectorPixelWidth']
        ZeroPaddedLength = int(2 ** (ceil(log2(2 * (nx - 1)))))
    #         DetectorPixelWidth = 445.059 / nx;
        print(DetectorPixelWidth)
        recon = np.zeros(self.recon.shape)
        DetectorSize = nx * DetectorPixelWidth
        fov = 2 * R * sin(atan(DetectorSize / 2 / (D + R)))
        self.params['fov'] = fov
        x = np.linspace(-fov / 2, fov / 2, self.params['ReconX'])
        y = np.linspace(-fov / 2, fov / 2, self.params['ReconY'])
        [xx, yy] = np.meshgrid(x, y)
        [phi, rho] = FanBeam.cart2pol(xx, yy)
        ReconZ = self.recon.shape[2]
        ProjectionAngle = np.linspace(0, self.params['AngleCoverage'], ns + 1)
        ProjectionAngle = ProjectionAngle[0:-1]
        dtheta = ProjectionAngle[1] - ProjectionAngle[0]
        assert(len(ProjectionAngle == ns))
        print('Reconstruction starts')
        cutoff = 0.3
        FilterType = 'hann'
        filter = FanBeam.Filter(
            ZeroPaddedLength + 1, DetectorPixelWidth * (R / (R + D)), FilterType, cutoff)
        gamma = np.arange(0, nx) - (nx - 1) / 2.0
        gamma = gamma * DetectorPixelWidth * (R / (R + D))
    #         for j in range(0, ny):
        for j in range(0, ny):
            sino = self.sino[:, :, j]
            weight = R / (sqrt(R ** 2 + gamma ** 2))
            for i in range(0, ns):
                angle = ProjectionAngle[i]
                WeightedSino = weight * sino[i, :]
                Q = real(
                    ifft(ifftshift(filter * fftshift(fft(WeightedSino, ZeroPaddedLength)))))
                t = xx * cos(angle) + yy * sin(angle)
                s = -xx * sin(angle) + yy * cos(angle)
                InterpX = (R * t) / (R - s)
        #                 print(InterpX.max(), InterpX.min(), R)
                U = (D + rho * sin(angle - phi)) / D
                f = interp1d(gamma, Q, kind='linear',
                             bounds_error=False, fill_value=0)
                vq = f(InterpX)
                recon[:, :, j] += dtheta * (1 / (U ** 2)) * vq.reshape(
                    [self.params['ReconX'], self.params['ReconY']])

        '''
                TO DO: Write file name definition
                       Save reconstruction condition

                '''
        self.recon = recon.astype(np.float32)
        self.recon.tofile(savefile, sep='', format='')
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
                    self.sino = self.proj.T
    #                     for i in range(ny):
    #                         self.sino[c, :, i] = image[i,:]
                c += 1
        except ErrorDescription as e:
            print(e)

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

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (phi, rho)


def main():
    start_time = time.time()
    filename = './ReconstructionParamsFan.txt'
    R = FanBeam(filename)
    R.LoadData()
    R.Reconstruction('./Recon.dat')
    print('%f seconds taken\n' % (time.time() - start_time))
    plt.imshow(R.recon[:, :, 99], cmap='gray',
               vmin=R.recon.min(), vmax=R.recon.max())
    plt.show()


if __name__ == '__main__':
    main()
