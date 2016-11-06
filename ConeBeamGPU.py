import os
import numpy as np
from scipy.interpolate import interp2d, griddata
import glob
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy.matlib
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
# function alias ends

# GPU function definition starts
mod = SourceModule("""
#include<stdio.h>
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

__global__ void gpuInterpol2d(float* Dest, float* check,float* codomain,float* domainX,float* domainY,float* new_domainX ,float* new_domainY,float* params){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    //printf("%d %d \\n",x,y);
    int NewDomainLengthX=params[6],NewDomainLengthY=params[7];
    int tid=x+y*(NewDomainLengthX*NewDomainLengthX);
    float M,N,S;
    
    if( tid<NewDomainLengthX*NewDomainLengthY*NewDomainLengthX*NewDomainLengthY){
        int OrgDomainLengthX=params[4],OrgDomainLengthY=params[5];
        float NewX=new_domainX[tid];
        float NewY=new_domainY[tid];
        check[tid]=1;
        float OrgDomain0X=params[0];
        float dOrgDomainX=params[1];
        float OrgDomain0Y=params[2];
        float dOrgDomainY=params[3];
        float fill_value=params[8];
        int XLow=floor((NewX-OrgDomain0X)/dOrgDomainX);
        int XHigh=ceil((NewX-OrgDomain0X)/dOrgDomainX);
        int YLow=floor((NewY-OrgDomain0Y)/dOrgDomainY);
        int YHigh=ceil((NewY-OrgDomain0Y)/dOrgDomainY);
        //printf("%f %f \\n",NewX,NewY);
        //printf("%d %d %d \\n",x,y,tid);
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
            if(isnan(M) || isnan(N)||isnan(S)){
                printf("%d %f %f %f %f %f %f %f %d %d \\n",tid, codomain[XLow+YLow*OrgDomainLengthX],w1/(w1+w2), w1 ,w2 ,w3 ,w4,NewX,XHigh,XLow);
            }
            //Save interpolated value to selected index
            Dest[tid]=S;
        }else{
            Dest[tid]=fill_value;
        }
    }else{
        //printf(" %d %d \\n",x,y);
    }
}

""")
# GPU function definition ends

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
        self.params = {'SourceToDetector':0, 'SourceToAxis':0, 'DataPath':'',
                     'precision':'', 'AngleCoverage':0, 'ReconX':0, 'ReconY':0,
                     'ReconZ':0, 'DetectorPixelHeight':0, 'DetectorPixelWidth':0,
                     'DetectorWidth':0, 'DetectorHeight':0, 'NumberOfViews':0,
                     'fov':0, 'fovz':0, 'Mode':0}
        f = open(filename)
        try:
            while True:
                line = f.readline()
                if not line: break
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
        self.recon = np.zeros([self.params['ReconX'], self.params['ReconY'], self.params['ReconZ']])

    def Reconstruction(self, savefile):
        R = self.params['SourceToAxis']
        D = self.params['SourceToDetector'] - R
        nx = int(self.params['DetectorWidth'])
        ny = int(self.params['DetectorHeight'])
        ns = int(self.params['NumberOfViews'])
        DetectorPixelWidth = self.params['DetectorPixelWidth']
        DetectorPixelHeight = self.params['DetectorPixelHeight']
        recon = np.zeros([self.params['ReconX'], self.params['ReconY'], self.params['ReconZ']])
        DetectorSize = [nx * DetectorPixelWidth, ny * DetectorPixelHeight]
        fov = 2 * R * sin(atan(DetectorSize[0] / 2 / (D + R)))
        fovz = 2 * R * sin(atan(DetectorSize[1] / 2 / (D + R)))
        self.params['fov'] = fov 
        self.params['fovz'] = fovz
        x = np.linspace(-fov / 2, fov / 2, self.params['ReconX'])
        y = np.linspace(-fov / 2, fov / 2, self.params['ReconY'])
        z = np.linspace(-fovz / 2, fovz / 2, self.params['ReconZ'])
        [xx, yy] = np.meshgrid(x, y)
        ReconZ = self.params['ReconZ']
        ProjectionAngle = np.linspace(0, self.params['AngleCoverage'], ns + 1)
        ProjectionAngle = ProjectionAngle[0:-1]
        dtheta = ProjectionAngle[1] - ProjectionAngle[0]
        fill_value = 0
        assert(len(ProjectionAngle == ns))
        print('Reconstruction starts')
        # ki = np.arange(0 - (nx - 1) / 2, nx - (nx - 1) / 2)
        # p = np.arange(0 - (ny - 1) / 2, ny - (ny - 1) / 2)
        ki = np.arange(0, nx) - (nx - 1) / 2
        p = np.arange(0, ny) - (ny - 1) / 2
        ki = ki * DetectorPixelWidth
        p = p * DetectorPixelHeight
        h = ConeBeam.Filter(nx, DetectorPixelWidth * R / (D + R), 0.5, 0.3)
        ki = (ki * R) / (R + D)
        p = (p * R) / (R + D)
        filter = np.absolute(fftshift(fft(h)))
        # filter = fftshift(fft(h))
        [kk, pp] = np.meshgrid(ki, p)
        interp2d_gpu = mod.get_function("gpuInterpol2d")
        sample_points = np.vstack((kk.flatten(), pp.flatten())).T
        weight = R / (sqrt(R ** 2 + kk ** 2 + pp ** 2))
        OrgDomain0X = ki.min()
        OrgDomain0Y = p.min()
        dOrgDomainX = ki[1] - ki[0]
        dOrgDomainY = p[1] - p[0]
        ki = ki.astype(np.float32)
        p = p.astype(np.float32)
        # Interpolgpu=mod.get_function("gpuInterpolLinear")
        for i in range(0, ns):
            angle = ProjectionAngle[i]
            print(i)
            WeightedProjection = weight * self.proj[:, :, i]
            Q = np.zeros(WeightedProjection.shape)
            for k in range(ny):
                Q[k, :] = real(ifft(ifftshift(filter * fftshift(fft(WeightedProjection[k, :])))))
            t = xx * cos(angle) + yy * sin(angle)
            s = -xx * sin(angle) + yy * cos(angle)
            # for l in range(ReconZ):
            InterpX = (R * t) / (R - s)
            InterpX2 = np.squeeze(repmat(InterpX.flatten(), 1, ReconZ)).astype(np.float32)
            zz = np.repeat(z, s.shape[0] * s.shape[1]).reshape([s.shape[0], s.shape[1], ReconZ])
            InterpY = (R * zz) / (R - s)
            InterpY2 = InterpY.flatten().astype(np.float32)
            print(InterpX2.shape, InterpY2.shape)
            assert(InterpX2.shape == InterpY2.shape)
            InterpW = (R ** 2) / ((R - s) ** 2)
            InterpW2 = repmat(InterpW.flatten(), 1, ReconZ)
            if(len(InterpX) * len(InterpY) < 512):
                blockX = len(InterpX)
                blockY = len(InterpY)
                blockZ = 1
                gridX = 1
                gridY = 1
            else:
                blockX = 32
                blockY = 32
                blockZ = 1
                gridX = 512 * 512 / (blockX * 2)  # int(np.ceil(len(InterpX2) * 1.0 / (blockX * blockY))) / 2
                gridY = 512 * 512 / (blockY * 2)  # int(np.ceil(len(InterpY2) * 1.0 / (blockX * blockY))) / 2
            print(blockX, blockY, gridX, gridY)
            dest = np.zeros([self.params['ReconX'], self.params['ReconY'] , ReconZ]).flatten().astype(np.float32)
            InterpParam = np.array([OrgDomain0X, dOrgDomainX, OrgDomain0Y, dOrgDomainY,
                    len(ki), len(p), 512, 512, fill_value]).astype(np.float32)
            Q = Q.flatten().astype(np.float32)
            IndexCheck = np.zeros_like(dest)
            interp2d_gpu(drv.Out(dest), drv.InOut(IndexCheck), drv.In(Q), drv.In(ki), drv.In(p),
                         drv.In(InterpX2), drv.In(InterpY2), drv.In(InterpParam),
                         block=(blockX, blockY, blockZ), grid=(gridX, gridY))
            assert((IndexCheck == 1).all())
            # print(np.isnan(dest).any(), np.isnan(dest).all())
            assert(not np.isnan(dest).any())
            vq = dest * InterpW2 * dtheta
            vq = vq.reshape([self.params['ReconX'], self.params['ReconY'], ReconZ])
            del dest
            recon += InterpW * dtheta * vq
#             for l in range(0, ReconZ):
#                 InterpX = (R * t) / (R - s)
#                 InterpY = (R * z[l]) / (R - s)
#                 InterpW = (R ** 2) / ((R - s) ** 2)
#                 InterpX2 = InterpX.flatten().astype(np.float32)
#                 InterpY2 = InterpY.flatten().astype(np.float32)
#                 np.testing.assert_almost_equal(InterpX2, InterpX.flatten(), 5)
#                 np.testing.assert_almost_equal(InterpY2, InterpY.flatten(), 5)
                # print((InterpX[:][0]==InterpX[:][1]).all())
                # InterpX = InterpX[0][:].astype(np.float32)
                # InterpY = InterpY[0][:].astype(np.float32)
#                 if(len(InterpX) * len(InterpY) < 512):
#                     blockX = len(InterpX)
#                     blockY = len(InterpY)
#                     blockZ = 1
#                     gridX = 1
#                     gridY = 1
#                 else:
#                     blockX = 32
#                     blockY = 32
#                     blockZ = 1
#                     gridX = int(np.ceil(len(InterpX) * 1.0 / (blockX * blockY))) / 2
#                     gridY = int(np.ceil(len(InterpX) * 1.0 / (blockX * blockY))) / 2
# #                     gridY = int(np.ceil(len(InterpY) * 1.0 / blockY))
#                     
#                 dest = np.zeros([self.params['ReconX'], self.params['ReconY']]).flatten().astype(np.float32)
#                 InterpParam = np.array([OrgDomain0X, dOrgDomainX, OrgDomain0Y, dOrgDomainY,
#                         len(ki), len(p), 512, 512, fill_value]).astype(np.float32)
#                 Q = Q.flatten().astype(np.float32)
#                 IndexCheck = np.zeros_like(dest)
# #                 vq = griddata(sample_points, Q.flatten(), (InterpY, InterpX), method='linear', fill_value=0)
#                 interp2d_gpu(drv.Out(dest), drv.InOut(IndexCheck), drv.In(Q), drv.In(ki), drv.In(p),
#                              drv.In(InterpX2), drv.In(InterpY2), drv.In(InterpParam),
#                              block=(blockX, blockY, blockZ), grid=(gridX, gridY))
#                 assert((IndexCheck == 1).all())
#                 # print(np.isnan(dest).any(), np.isnan(dest).all())
#                 assert(not np.isnan(dest).any())
#                 vq = dest.reshape([self.params['ReconX'], self.params['ReconY']])
#                 del dest
#                 
#                 recon[:, :, l] += InterpW * dtheta * vq
            # Interpolation required
        
        '''
        TO DO: Write file name definition
               Save reconstruction condition
               
        '''
        self.recon = recon
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
            
    @staticmethod
    def Filter(N, pixel_size, smooth, cutoff):
        ''' 
        TO DO: Ram-Lak filter implementation
               Argument for name of filter
        '''
        try:
            if cutoff > 0.5 or cutoff < 0:
                raise ErrorDescription(4)
            if smooth > 1.0 or smooth < 0:
                raise ErrorDescription(5)
        except ErrorDescription as e:
            print(e)
        EvenFlag = 0
        if(N % 2 == 0):
            N += 1
            EvenFlag = 1
        x = np.arange(1, N + 1) - (N - 1) / 2
        fm = cutoff / pixel_size
        x1 = x
        q1 = x1
        q1[np.where(x1 == 0)] = 0.01
        x2 = x - 0.5 / cutoff
        q2 = x2
        q2[np.where(x2 == 0)] = 0.01
        x3 = x + 0.5 / cutoff
        q3 = x3
        q3[np.where(x3 == 0)] = 0.01
        h1 = ((2 * fm ** 2) * sinc(2 * fm * pixel_size * q1)) - ((fm ** 2) * sinc(fm * pixel_size * q1) ** 2)
        h2 = ((2 * fm ** 2) * sinc(2 * fm * pixel_size * q2)) - ((fm ** 2) * sinc(fm * pixel_size * q2) ** 2)
        h3 = ((2 * fm ** 2) * sinc(2 * fm * pixel_size * q3)) - ((fm ** 2) * sinc(fm * pixel_size * q3) ** 2)
        h1[np.where(x1 == 0)] = fm ** 2
        h2[np.where(x2 == 0)] = fm ** 2
        h3[np.where(x3 == 0)] = fm ** 2
        h = smooth * h1 + (1 - smooth) / 2 * (h2 + h3)
        if(EvenFlag):
            h = h[0:-1]
            assert(h.shape[0] == N - 1)
        return h
    
def main():
    filename = './ReconstructionParams.txt'
    R = ConeBeam(filename)
    R.LoadData()
    R.Reconstruction('./Recon.dat')
    print(R.recon.min(), R.recon.max())
    plt.imshow(R.recon[:, :, 255], cmap='gray', vmin=R.recon.min(), vmax=R.recon.max())
    plt.show()

if __name__ == '__main__':
    main()
