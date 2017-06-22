import os
import sys
import numpy as np
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
repmat = numpy.matlib.repmat
ceil = np.ceil
log2 = np.log2
pi = np.pi
# function alias ends

# GPU function definition starts
mod = SourceModule("""
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_atomic_functions.h"

__global__ void gpuInterpol1d(float* Dest, float* codomain,float* domain ,float* new_domain,float* params){
    int x=threadIdx.x;
    float M;
    float NewX=new_domain[x];
    float OrgDomain0=params[0];
    float dOrgDomain=params[1];
    float ImgMin=params[6];
    int NewDomainLength=params[5],OrgDomainLength=params[4];
    int XLow=floor((NewX-OrgDomain0)/dOrgDomain);
    int XHigh=XLow+1;
    float w1=NewX-domain[XLow];
    float w2=domain[XHigh]-NewX;
    if(w1==0 && w2==0)
        w1=1.0;
    if(x<NewDomainLength){
        if(XHigh<OrgDomainLength && XLow>=0){
            //Bilinear interpolation for 2-D grid
            M=(w2/(w1+w2))*codomain[XLow]+(w1/(w1+w2))*codomain[XHigh];
            //Save interpolated value to selected index
            Dest[x]=M;
        }else{
            Dest[x]=ImgMin;
        }
    }
}

__global__ void Interpol2dLineargpu(float* Dest,float* codomain,float* domainX,float* domainY,float* new_domainX ,float* new_domainY,float* Weight,float* params){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int tid=y*gridDim.x*blockDim.x+x; //linear index
    int NewDomainLengthX=(int)params[6],NewDomainLengthY=(int)params[7],NewDomainLengthZ=(int)params[8];
    float dtheta=params[10];
    float M,N,S;

    if( tid<NewDomainLengthX*NewDomainLengthY*NewDomainLengthZ){
        int OrgDomainLengthX=(int)params[4],OrgDomainLengthY=(int)params[5];
        float NewX=new_domainX[tid];
        float NewY=new_domainY[tid];
        float OrgDomain0X=params[0];
        float dOrgDomainX=params[1];
        float OrgDomain0Y=params[2];
        float dOrgDomainY=params[3];
        float fill_value=params[9];
        int XLow=floor((NewX-OrgDomain0X)/dOrgDomainX);
        int XHigh=ceil((NewX-OrgDomain0X)/dOrgDomainX);
        int YLow=floor((NewY-OrgDomain0Y)/dOrgDomainY);
        int YHigh=ceil((NewY-OrgDomain0Y)/dOrgDomainY);
        float w1=NewX-domainX[XLow];
        float w2=domainX[XHigh]-NewX;
        float w3=NewY-domainY[YLow];
        float w4=domainY[YHigh]-NewY;
        if(fabs(w1-0)<0.0001 && fabs(w2-0)<0.0001)
            w1=1.0;
        if(fabs(w3-0)<0.0001 && fabs(w4-0)<0.0001)
            w3=1.0;
        M=0;N=0;S=0;
        //printf("%f %f %f %f \\n",w1,w2,w3,w4);
        if(XHigh<OrgDomainLengthX && XLow>=0 && YHigh<OrgDomainLengthY && YLow>=0){
            //Bilinear interpolation for 2-D grid
            M=(w2/(w1+w2))*codomain[XLow+YLow*OrgDomainLengthX]+(w1/(w1+w2))*codomain[XHigh+YLow*OrgDomainLengthX];
            N=(w2/(w1+w2))*codomain[XLow+YHigh*OrgDomainLengthX]+(w1/(w1+w2))*codomain[XHigh+YHigh*OrgDomainLengthX];
            S=(w4/(w3+w4))*M+(w3/(w3+w4))*N;
            //if(isnan(M) || isnan(N)||isnan(S)){
            //   printf("%d %f %f %f %f %f %f %f %d %d \\n",tid, codomain[XLow+YLow*OrgDomainLengthX],w1/(w1+w2), w1 ,w2 ,w3 ,w4,NewX,XHigh,XLow);
            //}
            //Save interpolated value to selected index
            //Dest[tid]+=S*dtheta*Weight[tid];
            atomicAdd(&Dest[tid],S*dtheta*Weight[tid]);
        }else{
            //Dest[tid]+=fill_value;
            atomicAdd(&Dest[tid],fill_value);
        }
    }
}
__global__ void Interpol2dSplinegpu(float* Dest,float* codomain,float* domainX,float* domainY,float* new_domainX ,float* new_domainY,float* Weight,float* params){
}

__global__ void CalculateInterpolGrid(float* InterpX,float* InterpY,float* InterpW,float* tt,float* ss,float* zz,float* params){
        int x=blockDim.x*blockIdx.x+threadIdx.x;
        int y=blockIdx.y*blockDim.y+threadIdx.y;
        int tid=y*gridDim.x*blockDim.x+x; //linear index
        float R=params[0];
        int ReconX=(int)params[1], ReconY=(int) params[2],ReconZ=(int)params[3];
        int tx=(tid%(ReconX*ReconY))%(ReconX);
        int ty=(tid%(ReconX*ReconY))/(ReconX);
        int tz=tid/(ReconX*ReconY);
        if(tx<ReconX && ty<ReconY && tz<ReconZ){
                InterpX[tid]=(R*tt[ty*ReconX+tx])/(R-ss[ty*ReconX+tx]);
                InterpY[tid]=(R*zz[tz])/(R-ss[ty*ReconX+tx]);
                InterpW[tid]=(R*R)/((R-ss[ty*ReconX+tx])*(R-ss[ty*ReconX+tx]));
        }
}

""")
# GPU function definition ends
'''TO DOs: Helical reconstruction
           Add interpolation method
           Half-fan weighting
           Read filter type and cutoff from file
           Merge GPU version and CPU version
'''


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
        elif(value == 6):
            self.msg = 'Recon volume is too big for GPU'
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
                       'fov': 0, 'fovz': 0, 'Mode': 0}
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
        R = self.params['SourceToAxis']
        D = self.params['SourceToDetector'] - R
        nx = int(self.params['DetectorWidth'])
        ny = int(self.params['DetectorHeight'])
        ns = int(self.params['NumberOfViews'])
        ReconX = self.params['ReconX']
        ReconY = self.params['ReconY']
        ReconZ = self.params['ReconZ']
        DetectorPixelWidth = self.params['DetectorPixelWidth']
        DetectorPixelHeight = self.params['DetectorPixelHeight']
        recon = np.zeros([ReconX, ReconY, ReconZ], dtype=np.float32)
        DetectorSize = [nx * DetectorPixelWidth, ny * DetectorPixelHeight]
        fov = 2 * R * sin(atan(DetectorSize[0] / 2 / (D + R)))
        fovz = 2 * R * sin(atan(DetectorSize[1] / 2 / (D + R)))
        self.params['fov'] = fov
        self.params['fovz'] = fovz
        ZeroPaddedLength = int(2 ** (ceil(log2(2 * (nx - 1)))))
        x = np.linspace(-fov / 2, fov / 2, ReconX)
        y = np.linspace(-fov / 2, fov / 2, ReconY)
        z = np.linspace(-fovz / 2, fovz / 2, ReconZ)
        [xx, yy] = np.meshgrid(x, y)
        ProjectionAngle = np.linspace(0, self.params['AngleCoverage'], ns + 1)
        ProjectionAngle = ProjectionAngle[0:-1]
        dtheta = ProjectionAngle[1] - ProjectionAngle[0]
        fill_value = 0
        assert(len(ProjectionAngle == ns))
        print('Reconstruction starts')
        start_time = time.time()
        ki = np.arange(0, nx) - (nx - 1) / 2
        p = np.arange(0, ny) - (ny - 1) / 2
        ki = ki * DetectorPixelWidth
        p = p * DetectorPixelHeight
        FilterType = 'ram-lak'
        cutoff = 0.3
        filter = ConeBeam.Filter(
            ZeroPaddedLength + 1, DetectorPixelWidth * R / (D + R), FilterType, cutoff)
        ki = (ki * R) / (R + D)
        p = (p * R) / (R + D)
        [kk, pp] = np.meshgrid(ki, p)
        interp2d_gpu = mod.get_function("Interpol2dLineargpu")
        GridCalculation_gpu = mod.get_function("CalculateInterpolGrid")
        weight = R / (sqrt(R ** 2 + kk ** 2 + pp ** 2))
        OrgDomain0X = ki.min()
        OrgDomain0Y = p.min()
        dOrgDomainX = ki[1] - ki[0]
        dOrgDomainY = p[1] - p[0]
        ki = ki.astype(np.float32)
        p = p.astype(np.float32)
        device = drv.Device(0)
        attrs = device.get_attributes()
        MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
        MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
        TotalSize = recon.shape[0] * recon.shape[1] * recon.shape[2]
        if(TotalSize < MAX_THREAD_PER_BLOCK):
            blockX = recon.shape[0] * recon.shape[1] * recon.shape[2]
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
                if(GridSize < MAX_GRID_DIM_X):
                    [gridX, gridY] = ConeBeam.OptimalGrid(GridSize)
                else:
                    raise ErrorDescription(6)
            except ErrorDescription as e:
                print(e)
                sys.exit()

        print(blockX, blockY, gridX, gridY)
        dest = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
        ki_gpu = pycuda.gpuarray.to_gpu(ki)
        p_gpu = pycuda.gpuarray.to_gpu(p)
        InterpParam = np.array([OrgDomain0X, dOrgDomainX, OrgDomain0Y, dOrgDomainY,
                                len(ki), len(p), ReconX, ReconY, ReconZ, fill_value, dtheta]).astype(np.float32)
        InterpParamgpu = pycuda.gpuarray.to_gpu(InterpParam)
        GridParam = np.array([R, ReconX, ReconY, ReconZ]).astype(np.float32)
        GridParamgpu = pycuda.gpuarray.to_gpu(GridParam)
#                 zz = np.tile(z, (ReconX, ReconY, 1)).T
        zz_dev = pycuda.gpuarray.to_gpu(z.flatten().astype(np.float32))
        InterpXgpu = pycuda.gpuarray.zeros_like(dest)
        InterpYgpu = pycuda.gpuarray.zeros_like(dest)
        InterpWgpu = pycuda.gpuarray.zeros_like(dest)
#                 InterpXgpu = np.zeros([ReconX * ReconY * ReconZ, 1], dtype=np.float32)
#                 InterpYgpu = np.zeros([ReconX * ReconY * ReconZ, 1], dtype=np.float32)
#                 InterpWgpu = np.zeros([ReconX * ReconY * ReconZ, 1], dtype=np.float32)
        start_time = time.time()
        for i in range(0, ns):
            angle = ProjectionAngle[i]
            print(i)
            WeightedProjection = weight * np.fliplr(self.proj[:, :, i])
            Q = np.zeros(WeightedProjection.shape)
            for k in range(ny):
                tmp = real(ifft(
                    ifftshift(filter * fftshift(fft(WeightedProjection[k, :], ZeroPaddedLength)))))
                Q[k, :] = tmp[0:nx]
            t = xx * cos(angle) + yy * sin(angle)
            s = -xx * sin(angle) + yy * cos(angle)
            tt = t.flatten().astype(np.float32)
            ss = s.flatten().astype(np.float32)
            GridCalculation_gpu(InterpXgpu, InterpYgpu, InterpWgpu, drv.In(tt),
                                drv.In(ss), zz_dev, GridParamgpu,
                                block=(blockX, blockY, blockZ), grid=(gridX, gridY))
            Q = Q.flatten().astype(np.float32)
            interp2d_gpu(dest, drv.In(Q), ki_gpu, p_gpu,
                         InterpXgpu, InterpYgpu, InterpWgpu, InterpParamgpu,
                         block=(blockX, blockY, blockZ), grid=(gridX, gridY))
            '''
                        TO DO: Save reconstruction condition
                        '''
        del InterpXgpu, InterpYgpu, InterpWgpu, zz_dev, InterpParamgpu, GridParamgpu
        recon = dest.get().reshape([ReconX, ReconY, ReconZ])
        self.recon = recon.astype(np.float32)
        recon.tofile(savefile, sep='', format='')

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
                print(len(filelist))
                raise ErrorDescription(3)
            else:
                c = 0
                for f in filelist:
                    image = np.fromfile(f, dtype=precision).reshape([ny, nx])
                    self.proj[:, :, c] = image
                    c += 1
        except ErrorDescription as e:
            print(e)
            sys.exit()

    @staticmethod
    def Filter(N, pixel_size, FilterType, cutoff):
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
    def OptimalGrid(GridSize):

        if(sqrt(GridSize).is_integer()):
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


def main():
    start_time = time.time()
    filename = './ReconstructionParamsMV.txt'
    R = ConeBeam(filename)
    R.LoadData()
    R.Reconstruction('./ReconMVRMI.dat')
    print(R.recon.min(), R.recon.max())
    print time.time() - start_time
#         plt.imshow(R.recon[:, :, 255], cmap='gray', vmin=R.recon.min(), vmax=R.recon.max())
#         plt.show()


if __name__ == '__main__':
    main()
