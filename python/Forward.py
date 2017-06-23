import numpy as np
import time
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray
from math import ceil


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
# function alias ends

# GPU function definition starts
mod = SourceModule("""
# include <stdio.h>
# include "cuda_runtime.h"
# include "device_launch_parameters.h"
# include "device_functions.h"
# include "device_atomic_functions.h"

__global__ void gpuInterpol1d(float* Dest, float* Xplane,float* Yplan,float* Zplane, float* DetectorIndex,float* params){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
	int tid=y*gridDim.x*blockDim.x+x; //linear index
	float alpha_x=Xplane/DetectorIndexX[tid]
	float alpha_y=Yplane/DetectorIndexY[tid]
	alpha_z=Zplane/DetectorIndexZ[tid]

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


class Forward:
    def __init__(self, filename, params):
        self.parms = {'SourceInit': [0, 0, 0], 'DetectorInit': [0, 0, 0], 'StartAngle': 0,
                      'EndAngle': 0, 'NumberOfDetectorPixels': [0, 0], 'DetectorPixelSize': [0, 0],
                      'NumberOfViews': 0, 'ImagePixelSpacing': [0, 0, 0], 'NumberOfImage': [0, 0, 0],
                      'PhantomCenter': [0, 0, 0]}

    def distance(self):
        SourceToDetector = self.parms['SourceToDetector']
        SourceToAxis = self.parms['SourceToAxis']
        NumberOfViews = self.parms['NumberOfViews']
        AngleCoverage = self.parms['AngleCoverage']
        DetectorChannels = self.parms['DetectorChannels']
        DetectorRows = self.parms['DetectorRows']
        DetectorPixelHeight = self.parms['DetectorPixelHeight']
        DetectorPixelWidth = self.parms['DetectorPixelWidth']
        # Calculates detector center
        angle = np.linspace(0, AngleCoverage, NumberOfViews + 1)
        for i in range(NumberOfViews):
            DetectorCenter = []
            CellCenters = []
            # calculates detector center for each view
            # calculates center of each cell
            # caculates boundary coordinates of each cell
            # create line for each boundary to source coordinate
            # difference plane for condition
            for j in range(DetectorChannels):
                for k in range(DetectorRows):
                    pass

    def ray_siddons(self):
        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [dv, du] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        Source_Init = self.params['SourceInit']
        Detector_Init = self.params['DetectorInit']
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = self.params['Origin']
        PhantomCenter = self.params['PhantomCenter']
		gpu = self.params['GPU']
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2))
        theta = np.linspace(StartAngle, EndAngle, nViews + 1)
        theta = theta[0:-2]
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
			elif(tan(angle)>=tol_max):
				DetectorIndex=[DetectorY+DetectorLengthU]
				DetectorIndexZ=DetectorZ-DetectorLengthV
			else:
				xx=sqrt(DetectorLengthU**2/(1+tan(angle)**2))
				yy=tan(angle)*sqrt(DetectorLengthU**2/(1+tan(angle)**2))
				DetectorIndex=[DetectorX*np.sign(DetectorLengthU*xx),]
			if(DetectorY>0):
				DetectorIndex=DetectoIndex[:,]
			DetectorIndex=DetectorIndex[:,1:-2]
			DetectorIndexZ=DetectorIndexZ[1:-2]
			if(gpu):

			else:


        if(save):
            proj.tofile(write_filename, sep='', format='')

        return proj


def main():
    start_time = time.time()
    filename = ''
    F = Forward(filename)
    end_time = time.time()


if __name__ == '__main__':
    main()
