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
# function alias starts
sin = np.sin
cos = np.cos
atan = np.arctan
tan = np.tan
sinc = np.sinc
sqrt = np.sqrt
repmat = numpy.matlib.repmat
ceil = np.ceil
pi = np.pi
floor = np.floor
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

__global__ void distance_project_on_y(float* Dest,float* Src,float* CoordX1,float* CoordX2,float* CoordZ1,float* CoordZ2,float* Xplane,float* Zplane,float* Xindex1,float* Xindex2,float* Zindex1,float* Zindex2,float* param){
        int x=blockDim.x*blockIdx.x+threadIdx.x;
        int y=blockDim.y*blockIdx.y+threadIdx.y;
        int tid=y*gridDim.x*blockDim.x+x;
        float dx=param[0];
        float dz=param[1];
        int nx=(int)param[2],ny=(int)param[3],nz=(int)param[4],nu=(int)param[5],nv=(int)param[6],iy=(int)param[7];
        int k=0,l=0,N=nu*nv;
        float weight1=0.0,weight2=0.0;
        int s_index_x=min((int)Xindex1[tid],(int)Xindex2[tid]);
        int e_index_x=max((int)Xindex1[tid],(int)Xindex2[tid]);
        int s_index_z=min((int)Zindex1[tid],(int)Zindex2[tid]);
        int e_index_z=max((int)Zindex1[tid],(int)Zindex2[tid]);
        if(tid<N){
            for(k=s_index_x;k<=e_index_x;k++){
                if(k>=0 && k<= nx-1){
                    if(s_index_x==e_index_x){
                        weight1=1.0;
                    }else if(k==s_index_x){
                        weight1=(Xplane[k+1]-fmin(CoordX1[tid],CoordX2[tid]))/fabs(CoordX1[tid]-CoordX2[tid]);
                    }else if(k==e_index_x){
                        weight1=(fmax(CoordX1[tid],CoordX2[tid])-Xplane[k])/fabs(CoordX1[tid]-CoordX2[tid]);
                    }else{
                        weight1=fabs(dx)/fabs(CoordX1[tid]-CoordX2[tid]);
                    }
                    if(fabs(weight1)<0.000001){
                        weight1=0.0;
                    }
                    for(l=s_index_z;l<=e_index_z;l++){
                        if(l>=0 && l<= nz-1){
                            if(s_index_z==e_index_z){
                                weight2=1.0;
                            }else if(l==s_index_z){
                                weight2=(fmax(CoordZ1[tid],CoordZ2[tid])-Zplane[l+1])/fabs(CoordZ1[tid]-CoordZ2[tid]);
                            }else if(l==e_index_z){
                                weight2=(Zplane[l]-fmin(CoordZ1[tid],CoordZ2[tid]))/fabs(CoordZ1[tid]-CoordZ2[tid]);
                            }else{
                                weight2=fabs(dz)/fabs(CoordZ1[tid]-CoordZ2[tid]);
                            }
                            if(fabs(weight2)<0.000001){
                                weight2=0.0;
                            }
                            atomicAdd(&Dest[tid],Src[(l*nx*ny)+iy*nx+k]*weight1*weight2);
                        }
                        
                        //syncthreads();
                    }
                }
            }
        }
}

__global__ void distance_project_on_x(float* Dest,float* Src,float* CoordY1,float* CoordY2,float* CoordZ1,float* CoordZ2,float* Yplane,float* Zplane,float* Yindex1,float* Yindex2,float* Zindex1,float* Zindex2,float* param){
        int x=blockDim.x*blockIdx.x+threadIdx.x;
        int y=blockDim.y*blockIdx.y+threadIdx.y;
        int tid=y*gridDim.x*blockDim.x+x;
        float dy=param[0];
        float dz=param[1];
        int nx=(int)param[2],ny=(int)param[3],nz=(int)param[4],nu=(int)param[5],nv=(int)param[6],ix=(int)param[7];
        int k=0,l=0,N=nu*nv;
        float weight1=0.0,weight2=0.0;
        int s_index_y=min((int)Yindex1[tid],(int)Yindex2[tid]);
        int e_index_y=max((int)Yindex1[tid],(int)Yindex2[tid]);
        int s_index_z=min((int)Zindex1[tid],(int)Zindex2[tid]);
        int e_index_z=max((int)Zindex1[tid],(int)Zindex2[tid]);
        if(tid<N){
            for(k=s_index_y;k<=e_index_y;k++){
                if(k>=0 && k<= ny-1){
                     if(s_index_y==e_index_y){
                        weight1=1.0;
                    }else if(k==s_index_y){
                        weight1=(fmax(CoordY1[tid],CoordY2[tid])-Yplane[k+1])/fabs(CoordY1[tid]-CoordY2[tid]);
                    }else if(k==e_index_y){
                        weight1=(Yplane[k]-fmin(CoordY1[tid],CoordY2[tid]))/fabs(CoordY1[tid]-CoordY2[tid]);
                    }else{
                        weight1=fabs(dy)/fabs(CoordY1[tid]-CoordY2[tid]);
                    }
                    if(fabs(weight1)<0.000001){
                        weight1=0.0;
                    }
                    for(l=s_index_z;l<=e_index_z;l++){
                        if(l>=0 && l<=nz-1){
                            if(s_index_z==e_index_z){
                                weight2=1.0;
                            }else if(l==s_index_z){
                                weight2=(fmax(CoordZ1[tid],CoordZ2[tid])-Zplane[l+1])/fabs(CoordZ1[tid]-CoordZ2[tid]);
                            }else if(l==e_index_z){
                                weight2=(Zplane[l]-fmin(CoordZ1[tid],CoordZ2[tid]))/fabs(CoordZ1[tid]-CoordZ2[tid]);
                            }else{
                                weight2=fabs(dz)/fabs(CoordZ1[tid]-CoordZ2[tid]);
                            }
                            if(fabs(weight2)<0.000001){
                                weight2=0.0;
                            }
                            atomicAdd(&Dest[tid],Src[(l*nx*ny)+k*nx+ix]*weight1*weight2);
                        }
                    }
                }
            }
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
        elif(value == 6):
            self.msg = 'Detector size is too big for GPU, try on CPU'
        else:
            self.msg = 'Unknown error'

    def __str__(self):
        return self.msg

class Forward:
    
    def __init__(self, filename, params):
        self.params = {'SourceInit': [0, 0, 0], 'DetectorInit': [0, 0, 0], 'StartAngle': 0,
                      'EndAngle': 0, 'NumberOfDetectorPixels': [0, 0], 'DetectorPixelSize': [0, 0],
                      'NumberOfViews': 0, 'ImagePixelSpacing': [0, 0, 0], 'NumberOfImage': [0, 0, 0],
                      'PhantomCenter': [0, 0, 0], 'Origina':[0, 0, 0], 'Method':'Distance', 'GPU':0}
        self.params = params
        start_time = time.time()
        self.image = self.LoadFile(filename, params['NumberOfImage'], dtype=np.float32)
        # print('File load: ' + str(time.time() - start_time))
    @staticmethod
    def LoadFile(filename, image_size, dtype=np.float32):
        image = np.fromfile(filename, dtype=dtype).reshape(image_size)
        return image
    def SaveProj(filename):
        self.proj.tofile(filename, sep='', format='')
    @staticmethod
    def DetectorConstruction(DetectorCenter, DetectorLength, DetectorVectors, angle):
        tol_min = 1e-5
        tol_max = 1e6
        # Do we need to use angle condition if detector construct based on direction vectors?
        # DetectorIndex=(DetectorCenter+DetectorLength[0]*DetectorVectors[0])+DetectorLength[1]*DetectorVectors[1]
        if(abs(tan(angle)) < tol_min):
            DetectorIndexX = DetectorCenter[0] + DetectorLength[0]
            DetectorIndexY = np.ones(DetectorLength[0].shape[0], dtype=np.float32) * DetectorCenter[1]
            DetectorIndexZ = DetectorCenter[2] - DetectorLength[1]
        elif(tan(angle) >= tol_max):
            DetectorIndexX = np.ones(DetectorLength[0].shape[0], dtype=np.float32) * DetectorCenter[0]
            DetectorIndexY = DetectorCenter[1] + DetectorLength[0]
            DetectorIndexZ = DetectorCenter[2] - DetectorLength[1]
        else:
            xx = sqrt(DetectorLength[0] ** 2 / (1 + tan(angle) ** 2))
            yy = tan(angle) * sqrt(DetectorLength[0] ** 2 / (1 + tan(angle) ** 2))
            DetectorIndexX = DetectorCenter[0] + np.sign(DetectorLength[0]) * xx
            DetectorIndexY = DetectorCenter[1] + np.sign(DetectorLength[0]) * yy 
            DetectorIndexZ = DetectorCenter[2] - DetectorLength[1]
#             plt.plot(xx)
#             plt.show()
        if(DetectorCenter[1] > 0):
            DetectorIndexX = DetectorIndexX[::-1]  # reverse the direction
            DetectorIndexY = DetectorIndexY[::-1]  # reverse the direction
        # print(DetectorIndexX.shape, DetectorIndexY.shape, DetectorIndexZ.shape)
        DetectorIndexX = DetectorIndexX[0:-1]
        DetectorIndexY = DetectorIndexY[0:-1]
        DetectorIndexZ = DetectorIndexZ[0:-1]
        # print(DetectorLength[0].shape, DetectorLength[1].shape)
        
        x = repmat(DetectorIndexX, DetectorLength[1].shape[0] - 1, 1)
        y = repmat(DetectorIndexY, DetectorLength[1].shape[0] - 1, 1)
        
        assert((x[0, :] == x[1, :]).all())
        assert((y[0, :] == y[1, :]).all())
        # z = np.reshape(repmat(DetectorIndexZ.T, 1, len(DetectorLength[0]) - 1), [len(DetectorLength[0]) - 1, len(DetectorLength[1]) - 1])
        z = repmat(DetectorIndexZ, DetectorLength[0].shape[0] - 1, 1).T
        assert((z[:, 0] == z[:, 1]).all())
        # print(x.shape, y.shape, z.shape)
#         plt.imshow(y,cmap='gray')
#         plt.show()
        DetectorIndex = np.vstack((x[np.newaxis], y[np.newaxis], z[np.newaxis]))
        return DetectorIndex
    @staticmethod
    def _optimalGrid(GridSize):

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
        gpu = self.params['GPU']
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2))
        # Calculates detector center
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = angle[0:-1]
        proj = np.zeros([nViews, nv, nu], dtype=np.float32)
        
        Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx + 1)) * dx
        Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny + 1)) * dy
        Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz + 1)) * dz
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
        print('Variable initialization: ' + str(time.time() - start_time))
        
        for i in range(nViews):
        #for i in range(12, 13):  
            print(i)
            start_time = time.time()
            Source = [-SAD * sin(angle[i]), SAD * cos(angle[i]), 0]  # z-direction rotation
            Detector = [(SDD - SAD) * sin(angle[i]), -(SDD - SAD) * cos(angle[i]), 0]
            DetectorLength = np.array([np.arange(floor(-nu / 2), floor(nu / 2) + 1) * du, np.arange(floor(-nv / 2), floor(nv / 2) + 1) * dv])
            DetectorVectors = [eu, ev, ew]
            DetectorIndex = self.DetectorConstruction(Detector, DetectorLength, DetectorVectors, angle[i])
            # print(DetectorIndex.shape, DetectorIndex.T.shape)
            DetectorIndex.tofile('DetectorIndex/%04d.dat' % i, sep='', format='')
            print('Detector initialization: ' + str(time.time() - start_time))
            if(self.params['Method'] == 'Distance'):
                start_time = time.time()
                proj[i, :, :] = self.distance(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)
                print('Total projection: ' + str(time.time() - start_time))
            elif(self.params['Method'] == 'Ray'):
                proj[i, :, :] = self.ray(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)
            print('time taken: ' + str(time.time() - start_time) + '\n')
                # proj[i,:,:]=distance_gpu()
        self.proj = proj    
                        
    def distance(self, DetectorIndex, Source, Detector, angle, Xplane, Yplane, Zplane):
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        dy = -1 * dy
        dz = -1 * dz
        proj = np.zeros([nv, nu], dtype=np.float32)
        if self.params['GPU']:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            TotalSize = nu * nv
            if(TotalSize < MAX_THREAD_PER_BLOCK):
                blockX = nu * nv
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
                        [gridX, gridY] = Forward._optimalGrid(GridSize)
                    else:
                        raise ErrorDescription(6)
                except ErrorDescription as e:
                    print(e)
                    sys.exit()
            distance_proj_on_y_gpu = mod.get_function("distance_project_on_y")
            distance_proj_on_x_gpu = mod.get_function("distance_project_on_x")
            image = self.image.flatten().astype(np.float32)
            dest = pycuda.gpuarray.to_gpu(proj.flatten().astype(np.float32))
            x_plane_gpu = pycuda.gpuarray.to_gpu(Xplane.astype(np.float32))
            y_plane_gpu = pycuda.gpuarray.to_gpu(Yplane.astype(np.float32))
            z_plane_gpu = pycuda.gpuarray.to_gpu(Zplane.astype(np.float32))
        start_time = time.time()
        DetectorBoundaryU1 = np.array([DetectorIndex[0, :, :] - cos(angle) * du / 2, DetectorIndex[1, :, :] - sin(angle) * du / 2, DetectorIndex[2, :, :]])
        DetectorBoundaryU2 = np.array([DetectorIndex[0, :, :] + cos(angle) * du / 2, DetectorIndex[1, :, :] + sin(angle) * du / 2, DetectorIndex[2, :, :]])
        DetectorBoundaryV1 = np.array([DetectorIndex[0, :, :] , DetectorIndex[1, :, :] , DetectorIndex[2, :, :] - dv / 2])
        DetectorBoundaryV2 = np.array([DetectorIndex[0, :, :] , DetectorIndex[1, :, :] , DetectorIndex[2, :, :] + dv / 2])
        
        # print('Detector boundary calculation: ' + str(time.time() - start_time))
        if(abs(Source[0] - Detector[0]) >= abs(Source[1] - Detector[1]) and abs(Source[0] - Detector[0]) >= abs(Source[2] - Detector[2])):
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
            for ix in range(nx):
                CoordY1 = SlopesU1 * (Xplane[ix] + dx / 2) + InterceptsU1
                CoordY2 = SlopesU2 * (Xplane[ix] + dx / 2) + InterceptsU2
                CoordZ1 = SlopesV1 * (Xplane[ix] + dx / 2) + InterceptsV1
                CoordZ2 = SlopesV2 * (Xplane[ix] + dx / 2) + InterceptsV2
                image_y1 = floor((CoordY1 - Yplane[0] + 0) / dy)
                image_y2 = floor((CoordY2 - Yplane[0] + 0) / dy)
                image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                if(self.params['GPU']):
                    proj_param = np.array([dy, dz, nx, ny, nz, nu, nv, ix]).astype(np.float32)
                    coord_y1_gpu = pycuda.gpuarray.to_gpu(CoordY1.flatten().astype(np.float32))
                    coord_y2_gpu = pycuda.gpuarray.to_gpu(CoordY2.flatten().astype(np.float32))
                    coord_z1_gpu = pycuda.gpuarray.to_gpu(CoordZ1.flatten().astype(np.float32))
                    coord_z2_gpu = pycuda.gpuarray.to_gpu(CoordZ2.flatten().astype(np.float32))
                    image_y1_gpu = pycuda.gpuarray.to_gpu(image_y1.flatten().astype(np.float32))
                    image_y2_gpu = pycuda.gpuarray.to_gpu(image_y2.flatten().astype(np.float32))
                    image_z1_gpu = pycuda.gpuarray.to_gpu(image_z1.flatten().astype(np.float32))
                    image_z2_gpu = pycuda.gpuarray.to_gpu(image_z2.flatten().astype(np.float32))
                    proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)                    
                    distance_proj_on_x_gpu(dest, drv.In(image), coord_y1_gpu,
                                           coord_y2_gpu, coord_z1_gpu, coord_z2_gpu, y_plane_gpu, z_plane_gpu,
                                           image_y1_gpu, image_y2_gpu, image_z1_gpu, image_z2_gpu, proj_param_gpu,
                                           block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                    del coord_y1_gpu, coord_y2_gpu, coord_z1_gpu, coord_z2_gpu, image_y1_gpu, image_y2_gpu, image_z1_gpu, image_z2_gpu, proj_param_gpu
                else:
                    proj += self._distance_project_on_x(self.image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane, Zplane, image_y1, image_y2, image_z1, image_z2, dy, dz, ix) * intersection_length
            if(self.params['GPU']):
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * intersection_length
                del dest
                
                   
        elif(abs(Source[1] - Detector[1]) >= abs(Source[0] - Detector[0]) and abs(Source[1] - Detector[1]) >= abs(Source[2] - Detector[2])):
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
            for iy in range(ny):
                # print(iy)
                start_time = time.time()
                CoordX1 = SlopesU1 * (Yplane[iy] + dy / 2) + InterceptsU1
                CoordX2 = SlopesU2 * (Yplane[iy] + dy / 2) + InterceptsU2
                CoordZ1 = SlopesV1 * (Yplane[iy] + dy / 2) + InterceptsV1
                CoordZ2 = SlopesV2 * (Yplane[iy] + dy / 2) + InterceptsV2
                # print(dx,dz)
                image_x1 = floor((CoordX1 - Xplane[0] + 0) / dx)
                image_x2 = floor((CoordX2 - Xplane[0] + 0) / dx)
                image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                # print('Coordiate calculation: ' + str(time.time() - start_time))
                if(self.params['GPU']):
                    proj_param = np.array([dx, dz, nx, ny, nz, nu, nv, iy]).astype(np.float32)
                    coord_x1_gpu = pycuda.gpuarray.to_gpu(CoordX1.flatten().astype(np.float32))
                    coord_x2_gpu = pycuda.gpuarray.to_gpu(CoordX2.flatten().astype(np.float32))
                    coord_z1_gpu = pycuda.gpuarray.to_gpu(CoordZ1.flatten().astype(np.float32))
                    coord_z2_gpu = pycuda.gpuarray.to_gpu(CoordZ2.flatten().astype(np.float32))
                    image_x1_gpu = pycuda.gpuarray.to_gpu(image_x1.flatten().astype(np.float32))
                    image_x2_gpu = pycuda.gpuarray.to_gpu(image_x2.flatten().astype(np.float32))
                    image_z1_gpu = pycuda.gpuarray.to_gpu(image_z1.flatten().astype(np.float32))
                    image_z2_gpu = pycuda.gpuarray.to_gpu(image_z2.flatten().astype(np.float32))
                    proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                    distance_proj_on_y_gpu(dest, drv.In(image), coord_x1_gpu,
                                           coord_x2_gpu, coord_z1_gpu, coord_z2_gpu, x_plane_gpu, z_plane_gpu,
                                           image_x1_gpu, image_x2_gpu, image_z1_gpu, image_z2_gpu, proj_param_gpu,
                                           block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                     
                    del coord_x1_gpu, coord_x2_gpu, coord_z1_gpu, coord_z2_gpu, image_x1_gpu, image_x2_gpu, image_z1_gpu, image_z2_gpu, proj_param_gpu
                else:
                    proj += self._distance_project_on_y(self.image, CoordX1, CoordX2, CoordZ1, CoordZ2, Xplane, Zplane, image_x1, image_x2, image_z1, image_z2, dx, dz, iy) * intersection_length
            if(self.params['GPU']):
                proj = dest.get().reshape([nv, nu])
                proj = proj.astype(np.float32) * intersection_length
                del dest
                
        else:
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
            for iz in range(nz):
                CoordX1 = SlopesU1 * Zplane[iz] + dz / 2 + InterceptsU1
                CoordX2 = SlopesU2 * Zplane[iz] + dz / 2 + InterceptsU2
                CoordY1 = SlopesV1 * Zplane[iz] + dz / 2 + InterceptsV1
                CoordY2 = SlopesV2 * Zplane[iz] + dz / 2 + InterceptsV2
                image_x1 = floor(CoordX1 - Xplane[0] + dx) / dx
                image_x2 = floor(CoordX2 - Xplane[0] + dx) / dx
                image_y1 = floor(CoordY1 - Yplane[0] + dy) / dy
                image_y2 = floor(CoordY2 - Yplane[0] + dy) / dy
                if(self.params['GPU']):
                    pass
                else:
                    proj += self._distance_project_on_z(self.image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane, image_x1, image_x2, image_y1, image_y2, dx, dy, iz) * intersection_length
        return proj
    @staticmethod
    def _distance_project_on_y(image, CoordX1, CoordX2, CoordZ1, CoordZ2, Xplane, Zplane, image_x1, image_x2, image_z1, image_z2, dx, dz, iy):
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
                    if(k < 0 or k > image.shape[0] - 1):
                        continue
                    if(s_index_x == e_index_x):
                        weight1 = 1
                    elif(k == s_index_x):
                        # print(k,s_index_x,e_index_x,Xplane[k+1],CoordX1[i,j],CoordX2[i,j])
                        weight1 = (Xplane[k + 1] - min(CoordX1[i, j], CoordX2[i, j])) / abs(CoordX1[i, j] - CoordX2[i, j])
                    elif(k == e_index_x):
                        # print(k,s_index_x,e_index_x)
                        # print(Xplane[k],CoordX1[i,j],CoordX2[i,j])
                        weight1 = (max(CoordX1[i, j], CoordX2[i, j]) - Xplane[k]) / abs(CoordX1[i, j] - CoordX2[i, j])
                    else:
                        weight1 = abs(dx) / abs(CoordX1[i, j] - CoordX2[i, j])
                    for l in range(int(s_index_z), int(e_index_z) + 1):
                        if(l < 0 or l > image.shape[2] - 1):
                            continue
                        if(s_index_z == e_index_z):
                            weight2 = 1
                        elif(l == s_index_z):
                            # print(s_index_z,e_index_z,Zplane[l+1],CoordZ1[i,j],CoordZ2[i,j])
                            weight2 = (max(CoordZ1[i, j], CoordZ2[i, j]) - Zplane[l + 1]) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        elif(l == e_index_z):
                            # print('1')
                            weight2 = (Zplane[l] - min(CoordZ1[i, j], CoordZ2[i, j])) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        else:
                            weight2 = abs(dz) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        # print(weight1,weight2)
                        assert(weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[l][iy][k]
                proj[i, j] = p_value
        # print('Projection for a loop: ' + str(time.time() - start_time))
        return proj
    @staticmethod
    def _distance_project_on_x(image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane, Zplane, image_y1, image_y2, image_z1, image_z2, dy, dz, ix):
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
                    if(k < 0 or k > image.shape[1] - 1):
                        continue
                    if(s_index_y == e_index_y):
                        weight1 = 1
                    elif(k == s_index_y):
                        weight1 = (max(CoordY1[i, j], CoordY2[i, j]) - Yplane[k + 1]) / abs(CoordY1[i, j] - CoordY2[i, j])
                    elif(k == e_index_y):
                        weight1 = (Yplane[k ] - min(CoordY1[i, j], CoordY2[i, j])) / abs(CoordY1[i, j] - CoordY2[i, j])
                    else:
                        weight1 = abs(dy) / abs(CoordU1[i, j] - CoordU2[i, j])
                    #if(abs(weight1) - 0 < tol_min):
                    #    weight1 = 0
                    for l in range(int(s_index_z), int(e_index_z) + 1):
                        if(l < 0 or l > image.shape[2] - 1):
                            continue
                        if(s_index_z == e_index_z):
                            weight2 = 1
                        elif(l == s_index_z):
                            weight2 = (max(CoordZ1[i, j], CoordZ2[i, j]) - Zplane[l + 1]) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        elif(l == e_index_z):
                            weight2 = (Zplane[l] - min(CoordZ1[i, j], CoordZ2[i, j])) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        else:
                            weight2 = abs(dz) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        # print(s_index_z,e_index_z,Zplane[l+1],Zplane[l],CoordZ1[i,j],CoordZ2[i,j])
                        #if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        #print(weight1,weight2)
                        assert(weight1>0 and weight2>0 and weight1<=1 and weight2<=1)
                        p_value += weight1 * weight2 * image[l][k][ix]
                proj[i, j] = p_value
        return proj
                            
    @staticmethod
    def _distance_project_on_z(image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane, image_x1, image_X2, image_y1, image_y2, dx, dy, iz):
        pass
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
        angle = theta[0:-2]
        Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx)) * dx
        Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny)) * dy
        Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz)) * dz
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
            if(abs(tan(angle)) < tol_min):
                DetectorIndex = [DetectorX + DetectlrLengthU]
                DetectorIndexZ = DetectorZ - DetectorLengthV
            elif(tan(angle) >= tol_max):
                DetectorIndex = [DetectorY + DetectorLengthU]
                DetectorIndexZ = DetectorZ - DetectorLengthV
            else:
                xx = sqrt(DetectorLengthU ** 2 / (1 + tan(angle) ** 2))
                yy = tan(angle) * sqrt(DetectorLengthU ** 2 / (1 + tan(angle) ** 2))
                DetectorIndex = [DetectorX * np.sign(DetectorLengthU * xx), ]
            if(DetectorY > 0):
                DetectorIndex = DetectoIndex[:, ]
            DetectorIndex = DetectorIndex[:, 1:-2]
            DetectorIndexZ = DetectorIndexZ[1:-2]
            if(gpu):
                pass
            else:
                pass


        if(save):
            proj.tofile(write_filename, sep='', format='')

        return proj


def main():
    start_time = time.time()
    filename = 'Shepp_Logan_3d_256.dat'
    params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
                      'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.5, 0.5],
                      'NumberOfViews': 90, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
                      'PhantomCenter': [0, 0, 0], 'Origin':[0, 0, 0], 'Method':'Distance', 'GPU':1}
    F = Forward(filename, params)
    F.forward()
    F.proj.tofile('proj_distance.dat', sep='', format='')
    end_time = time.time()
    plt.imshow(F.proj[12, :, :], cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
