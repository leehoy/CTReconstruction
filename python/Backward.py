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
log2 = np.log2
pi = np.pi
floor = np.floor
fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
real = np.real

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


__global__ void distance_project_on_y2(float* Dest,float* Src,float* slope_x1,float* slope_x2,float* slope_z1,float* slope_z2,float* intercept_x1,float* intercept_x2,float* intercept_z1,float* intercept_z2,float* Xplane,float* Yplane,float* Zplane,float* param){
        int x=blockDim.x*blockIdx.x+threadIdx.x;
        int y=blockDim.y*blockIdx.y+threadIdx.y;
        int tid=y*gridDim.x*blockDim.x+x;
        float dx=param[0];
        float dy=param[1];
        float dz=param[2];
        int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7];
        int k=0,l=0,N=nu*nv;
        float weight1=0.0,weight2=0.0;
        int iy=(int) floor(tid*1.0/(N*1.0));
        int pix_num=tid-(N*iy); //floor(fmodf(tid*1.0,N*1.0));
        float coord_x1,coord_x2,coord_z1,coord_z2;
        int index_x1,index_x2,index_z1,index_z2;
        coord_x1=slope_x1[pix_num]*(Yplane[iy]+dy/2)+intercept_x1[pix_num];
        coord_x2=slope_x2[pix_num]*(Yplane[iy]+dy/2)+intercept_x2[pix_num];
        coord_z1=slope_z1[pix_num]*(Yplane[iy]+dy/2)+intercept_z1[pix_num];
        coord_z2=slope_z2[pix_num]*(Yplane[iy]+dy/2)+intercept_z2[pix_num];
        index_x1=(int) floor((coord_x1-Xplane[0])/dx);
        index_x2=(int) floor((coord_x2-Xplane[0])/dx);
        index_z1=(int) floor((coord_z1-Zplane[0])/dz);
        index_z2=(int) floor((coord_z2-Zplane[0])/dz);
        int s_index_x=min(index_x1,index_x2);
        int e_index_x=max(index_x1,index_x2);
        int s_index_z=min(index_z1,index_z2);
        int e_index_z=max(index_z1,index_z2);
        if(tid<N*ny){
            for(k=s_index_x;k<=e_index_x;k++){
                if(k>=0 && k<= nx-1){
                    if(s_index_x==e_index_x){
                        weight1=1.0;
                    }else if(k==s_index_x){
                        weight1=(Xplane[k+1]-fmin(coord_x1,coord_x2))/fabs(coord_x1-coord_x2);
                    }else if(k==e_index_x){
                        weight1=(fmax(coord_x1,coord_x2)-Xplane[k])/fabs(coord_x1-coord_x2);
                    }else{
                        weight1=fabs(dx)/fabs(coord_x1-coord_x2);
                    }
                    if(fabs(weight1)<0.000001){
                        weight1=0.0;
                    }
                    for(l=s_index_z;l<=e_index_z;l++){
                        if(l>=0 && l<= nz-1){
                            if(s_index_z==e_index_z){
                                weight2=1.0;
                            }else if(l==s_index_z){
                                weight2=(fmax(coord_z1,coord_z2)-Zplane[l+1])/fabs(coord_z1-coord_z2);
                            }else if(l==e_index_z){
                                weight2=(Zplane[l]-fmin(coord_z1,coord_z2))/fabs(coord_z1-coord_z2);
                            }else{
                                weight2=fabs(dz)/fabs(coord_z1-coord_z2);
                            }
                            if(fabs(weight2)<0.000001){
                                weight2=0.0;
                            }
                            atomicAdd(&Dest[pix_num],Src[(l*nx*ny)+iy*nx+k]*weight1*weight2);
                        }
                        
                        //syncthreads();
                    }
                }
            }
        }
}

__global__ void distance_project_on_x2(float* Dest,float* Src,float* slope_y1,float* slope_y2,float* slope_z1,float* slope_z2,float* intercept_y1,float* intercept_y2,float* intercept_z1,float* intercept_z2,float* Xplane,float* Yplane,float* Zplane,float* param){
        int x=blockDim.x*blockIdx.x+threadIdx.x;
        int y=blockDim.y*blockIdx.y+threadIdx.y;
        int tid=y*gridDim.x*blockDim.x+x;
        float dx=param[0];
        float dy=param[1];
        float dz=param[2];
        int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7];
        int k=0,l=0,N=nu*nv;
        float weight1=0.0,weight2=0.0;
        int ix=(int) floor(tid*1.0 /N);
        int pix_num=tid-(N*ix); //floor(fmodf(tid*1.0,N*1.0));
        float coord_y1,coord_y2,coord_z1,coord_z2;
        int index_y1,index_y2,index_z1,index_z2;
        coord_y1=slope_y1[pix_num]*(Xplane[ix]+dx/2)+intercept_y1[pix_num];
        coord_y2=slope_y2[pix_num]*(Xplane[ix]+dx/2)+intercept_y2[pix_num];
        coord_z1=slope_z1[pix_num]*(Xplane[ix]+dx/2)+intercept_z1[pix_num];
        coord_z2=slope_z2[pix_num]*(Xplane[ix]+dx/2)+intercept_z2[pix_num];
        index_y1=(int) floor((coord_y1-Yplane[0])*1.0/dy);
        index_y2=(int) floor((coord_y2-Yplane[0])*1.0/dy);
        index_z1=(int) floor((coord_z1-Zplane[0])*1.0/dz);
        index_z2=(int) floor((coord_z2-Zplane[0])*1.0/dz);
        int s_index_y=min(index_y1,index_y2);
        int e_index_y=max(index_y1,index_y2);
        int s_index_z=min(index_z1,index_z2);
        int e_index_z=max(index_z1,index_z2);
        if(tid<N*nx){
            for(k=s_index_y;k<=e_index_y;k++){
                if(k>=0 && k<= ny-1){
                     if(s_index_y==e_index_y){
                        weight1=1.0;
                    }else if(k==s_index_y){
                        weight1=(fmax(coord_y1,coord_y2)-Yplane[k+1])/fabs(coord_y1-coord_y2);
                    }else if(k==e_index_y){
                        weight1=(Yplane[k]-fmin(coord_y1,coord_y2))/fabs(coord_y1-coord_y2);
                    }else{
                        weight1=fabs(dy)/fabs(coord_y1-coord_y2);
                    }
                    if(fabs(weight1)<0.000001){
                        weight1=0.0;
                    }
                    for(l=s_index_z;l<=e_index_z;l++){
                        if(l>=0 && l<=nz-1){
                            if(s_index_z==e_index_z){
                                weight2=1.0;
                            }else if(l==s_index_z){
                                weight2=(fmax(coord_z1,coord_z2)-Zplane[l+1])/fabs(coord_z1-coord_z2);
                            }else if(l==e_index_z){
                                weight2=(Zplane[l]-fmin(coord_z1,coord_z2))/fabs(coord_z1-coord_z2);
                            }else{
                                weight2=fabs(dz)/fabs(coord_z1-coord_z2);
                            }
                            if(fabs(weight2)<0.000001){
                                weight2=0.0;
                            }
                            atomicAdd(&Dest[pix_num],Src[(l*nx*ny)+k*nx+ix]*weight1*weight2);
                        }
                    }
                }
            }
        }
}

__global__ void distance_project_on_z2(float* Dest,float* Src,float* slope_x1,float* slope_x2,float* slope_y1,float* slope_y2,float* intercept_x1,float* intercept_x2,float* intercept_y1,float* intercept_y2,float* Xplane,float* Yplane,float* Zplane,float* param){
        int x=blockDim.x*blockIdx.x+threadIdx.x;
        int y=blockDim.y*blockIdx.y+threadIdx.y;
        int tid=y*gridDim.x*blockDim.x+x;
        float dx=param[0];
        float dy=param[1];
        float dz=param[2];
        int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7];
        int k=0,l=0,N=nu*nv;
        float weight1=0.0,weight2=0.0;
        int iz=(int) floor(tid*1.0 /N);
        int pix_num=tid-(N*iz); 
        float coord_x1,coord_x2,coord_y1,coord_y2;
        int index_x1,index_x2,index_y1,index_y2;
        coord_x1=slope_x1[pix_num]*(Zplane[iz]+dz/2)+intercept_x1[pix_num];
        coord_x2=slope_x2[pix_num]*(Zplane[iz]+dz/2)+intercept_x2[pix_num];
        coord_y1=slope_y1[pix_num]*(Zplane[iz]+dz/2)+intercept_y1[pix_num];
        coord_y2=slope_y2[pix_num]*(Zplane[iz]+dz/2)+intercept_y2[pix_num];
        index_x1=(int) floor((coord_x1-Xplane[0])*1.0/dx);
        index_x2=(int) floor((coord_x2-Xplane[0])*1.0/dx);
        index_y1=(int) floor((coord_y1-Yplane[0])*1.0/dy);
        index_y2=(int) floor((coord_y2-Yplane[0])*1.0/dy);
        int s_index_x=min(index_x1,index_x2);
        int e_index_x=max(index_x1,index_x2);
        int s_index_y=min(index_y1,index_y2);
        int e_index_y=max(index_y1,index_y2);
        if(tid<N*nz){
            for(k=s_index_x;k<=e_index_x;k++){
                if(k>=0 && k<= nx-1){
                    if(s_index_x==e_index_x){
                        weight1=1.0;
                    }else if(k==s_index_x){
                        weight1=(Xplane[k+1]-fmin(coord_x1,coord_x2))/fabs(coord_x1-coord_x2);
                    }else if(k==e_index_x){
                        weight1=(fmax(coord_x1,coord_x2)-Yplane[k])/fabs(coord_x1-coord_x2);
                    }else{
                        weight1=fabs(dx)/fabs(coord_x1-coord_x2);
                    }
                    if(fabs(weight1)<0.000001){
                        weight1=0.0;
                    }
                    for(l=s_index_y;l<=e_index_y;l++){
                        if(l>=0 && l<=ny-1){
                            if(s_index_y==e_index_y){
                                weight2=1.0;
                            }else if(l==s_index_y){
                                weight2=(fmax(coord_y1,coord_y2)-Yplane[l+1])/fabs(coord_y1-coord_y2);
                            }else if(l==e_index_y){
                                weight2=(Yplane[l]-fmin(coord_y1,coord_y2))/fabs(coord_y1-coord_y2);
                            }else{
                                weight2=fabs(dy)/fabs(coord_y1-coord_y2);
                            }
                            if(fabs(weight2)<0.000001){
                                weight2=0.0;
                            }
                            atomicAdd(&Dest[pix_num],Src[(iz*nx*ny)+l*nx+k]*weight1*weight2);
                        }
                    }
                }
            }
        }
}

__global__ void distance_backproj_about_z(float* Dest, float* Src,float* x_plane,float* y_plane,float* z_plane,float* u_plane,float* v_plane,float* params){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int tid=y*gridDim.x*blockDim.x+x;
    float dx=params[0],dy=params[1],dz=params[2];
    int nx=(int)params[3],ny=(int)params[4],nz=(int)params[5],nu=(int)params[6],nv=(int)params[7];
    float du=params[8],dv=params[9];
    float SourceX=params[10],SourceY=params[11],SourceZ=params[12];
    float DetectorY=params[14];
    //float DetectorX=params[13],DetectorY=params[14],DetectorZ=params[15];
    float angle=params[16],R=params[17];
    //float xc,yc,zc;
    float yc;//,slope_tmp;
    float x1,x2,x3,x4,y1,y2,y3,y4,z1,z2;
    int k=0,l=0,N=nx*ny*nz,ix=0,iy=0,iz=0;
    float u_slope1,u_slope2,u_slope3,u_slope4;
    float v_slope1,v_slope2;//,v_slope3,v_slope4;
    float coord_u1,coord_u2,coord_v1,coord_v2;
    int index_u1,index_u2,index_v1,index_v2;
    float slope_min,slope_max;
    int s_index_u,e_index_u,s_index_v,e_index_v;
    float weight1,weight2,InterpWeight=0.0;
    if(tid<N){
        iz=(int)tid/(nx*ny*1.0);
        iy=(int)(tid-(iz*nx*ny))/(nx*1.0);
        ix=(int)(tid-iz*nx*ny-iy*nx);
        yc=-x_plane[ix]*sin(angle)+y_plane[iy]*cos(angle);
        x1=(x_plane[ix]+dx/2.0)*cos(angle)+(y_plane[iy]+dy/2.0)*sin(angle);
        y1=-(x_plane[ix]+dx/2.0)*sin(angle)+(y_plane[iy]+dy/2.0)*cos(angle);
        x2=(x_plane[ix]-dx/2.0)*cos(angle)+(y_plane[iy]-dy/2.0)*sin(angle);
        y2=-(x_plane[ix]-dx/2.0)*sin(angle)+(y_plane[iy]-dy/2.0)*cos(angle);
        x3=(x_plane[ix]+dx/2.0)*cos(angle)+(y_plane[iy]-dy/2.0)*sin(angle);
        y3=-(x_plane[ix]+dx/2.0)*sin(angle)+(y_plane[iy]-dy/2.0)*cos(angle);
        x4=(x_plane[ix]-dx/2.0)*cos(angle)+(y_plane[iy]+dy/2.0)*sin(angle);
        y4=-(x_plane[ix]-dx/2.0)*sin(angle)+(y_plane[iy]+dy/2.0)*cos(angle);
        z1=z_plane[iz]-dz/2.0;
        z2=z_plane[iz]+dz/2.0;
        u_slope1=(SourceX-x1)/(SourceY-y1);
        u_slope2=(SourceX-x2)/(SourceY-y2);
        u_slope3=(SourceX-x3)/(SourceY-y3);
        u_slope4=(SourceX-x4)/(SourceY-y4);
        //slope_tmp=fmin(u_slope1,u_slope2);
        //slope_tmp=fmin(slope_tmp,u_slope3);
        //slope_min=fmin(slope_tmp,u_slope4);
        //slope_tmp=fmax(u_slope1,u_slope2);
        //slope_tmp=fmax(slope_tmp,u_slope3);
        //slope_max=fmax(slope_tmp,u_slope4);
        slope_min=fmin(u_slope1,fmin(u_slope2,fmin(u_slope3,u_slope4)));
        slope_max=fmax(u_slope1,fmax(u_slope2,fmax(u_slope3,u_slope4)));
        coord_u1=(slope_min*DetectorY)+(SourceX-slope_min*SourceY);
        coord_u2=(slope_max*DetectorY)+(SourceX-slope_max*SourceY);
        index_u1=floor((coord_u1-u_plane[0])*1.0/du);
        index_u2=floor((coord_u2-u_plane[0])*1.0/du);
        s_index_u=min(index_u1,index_u2);
        e_index_u=max(index_u1,index_u2);
        v_slope1=(SourceZ-z1)/(SourceY-yc);
        v_slope2=(SourceZ-z2)/(SourceY-yc);
        slope_min=fmin(v_slope1,v_slope2);
        slope_max=fmax(v_slope1,v_slope2);
        coord_v1=(slope_min*DetectorY)+(SourceZ-slope_min*SourceY);
        coord_v2=(slope_max*DetectorY)+(SourceZ-slope_max*SourceY);
        index_v1=floor((coord_v1-v_plane[0])*1.0/dv);
        index_v2=floor((coord_v2-v_plane[0])*1.0/dv);
        s_index_v=min(index_v1,index_v2);
        e_index_v=max(index_v1,index_v2);
        InterpWeight=(R*R)/((R-yc)*(R-yc));
        for(k=s_index_v;k<=e_index_v;k++){
            if(k>=0 && k<=nv-1){
                if(s_index_v==e_index_v){
                    weight1=1.0;
                }else if(k==s_index_v){
                    weight1=(fmax(coord_v1,coord_v2)-v_plane[k+1])/fabs(coord_v1-coord_v2);
                }else if(k==e_index_v){
                    weight1=(v_plane[k]-fmin(coord_v1,coord_v2))/fabs(coord_v1-coord_v2);
                }else{
                    weight1=fabs(dv)/fabs(coord_v1-coord_v2);
                }
                for(l=s_index_u;l<=e_index_u;l++){
                    if(l>=0 && l<=nu-1){
                        if(s_index_u==e_index_u){
                            weight2=1.0;
                        }else if(l==s_index_u){
                            weight2=(u_plane[l+1]-fmin(coord_u1,coord_u2))/fabs(coord_u1-coord_u2);
                        }else if(l==e_index_u){
                            weight2=(fmax(coord_u1,coord_u2)-u_plane[l])/fabs(coord_u1-coord_u2);
                        }else{
                            weight2=fabs(du)/fabs(coord_u1-coord_u2);
                        }
                        atomicAdd(&Dest[tid],Src[k*nu+l]*weight1*weight2*InterpWeight);
                    }                    
                }
            }
        }
    }
}

__device__ float f_angle(float x,float y){
    float angle;
    if(y!=0){
        angle=atan2f(x,y);
    }else if(fabs(x)<0.000001){
        angle=0;
    }else if(x<0){
        angle=-3.14159265359/2.0;
    }else if(x>0){
        angle=3.14159265359/2.0;
    }
    return angle;
}

__device__ float fx(float x,float y,float z,float angle1,float angle2){
    float new_pos,angle;
    angle=f_angle(x,y);
    new_pos=x*cos(angle2)*cos(angle1)+y*cos(angle2)*sin(angle1)-z*sin(angle2)*cos(angle1)*sin(angle)-z*sin(angle2)*sin(angle1)*cos(angle);
    return new_pos;
}
__device__ float fy(float x,float y,float z,float angle1,float angle2){
    float new_pos,angle;
    angle=f_angle(x,y);
    new_pos=y*cos(angle2)*cos(angle1)-x*cos(angle2)*sin(angle1)-z*sin(angle2)*cos(angle1)*cos(angle)+z*sin(angle2)*sin(angle1)*sin(angle);
    return new_pos;
}
__device__ float fz(float x,float y,float z,float angle1,float angle2){
    float new_pos;
    new_pos=z*cos(angle2)+sqrtf(x*x+y*y)*sin(angle2);
    return new_pos;
}
__global__ void distance_backproj_arb(float* Dest, float* Src,float* x_plane,float* y_plane,float* z_plane,float* u_plane,float* v_plane,float* params){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int tid=y*gridDim.x*blockDim.x+x;
    float dx=params[0],dy=params[1],dz=params[2];
    int nx=(int)params[3],ny=(int)params[4],nz=(int)params[5],nu=(int)params[6],nv=(int)params[7];
    float du=params[8],dv=params[9];
    float SourceX=params[10],SourceY=params[11],SourceZ=params[12];
    float DetectorY=params[14];
    float angle1=params[16],angle2=params[17],R=params[18];
    float yc;//,slope_tmp;
    float x1,x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8,z1,z2,z3,z4,z5,z6,z7,z8;
    int k=0,l=0,N=nx*ny*nz,ix=0,iy=0,iz=0;
    float u_slope1,u_slope2,u_slope3,u_slope4,u_slope5,u_slope6,u_slope7,u_slope8;
    float v_slope1,v_slope2,v_slope3,v_slope4,v_slope5,v_slope6,v_slope7,v_slope8;
    float coord_u1,coord_u2,coord_v1,coord_v2;
    int index_u1,index_u2,index_v1,index_v2;
    float slope_min,slope_max;
    int s_index_u,e_index_u,s_index_v,e_index_v;
    float weight1,weight2,InterpWeight=0.0;
    if(tid<N){
        iz=(int)tid/(nx*ny*1.0);
        iy=(int)(tid-(iz*nx*ny))/(nx*1.0);
        ix=(int)(tid-iz*nx*ny-iy*nx);
        yc=fy(x_plane[ix],y_plane[iy],z_plane[iz],angle1,angle2);
        x1=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        y1=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        z1=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        
        x2=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        y2=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        z2=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        
        x3=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        y3=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        z3=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        
        x4=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        y4=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        z4=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2,angle1,angle2);
        
        x5=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        y5=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        z5=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        
        x6=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        y6=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        z6=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        
        x7=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        y7=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        z7=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        
        x8=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        y8=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2,angle1,angle2);
        z8=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2,angle1,angle2);

        u_slope1=(SourceX-x1)/(SourceY-y1);
        u_slope2=(SourceX-x2)/(SourceY-y2);
        u_slope3=(SourceX-x3)/(SourceY-y3);
        u_slope4=(SourceX-x4)/(SourceY-y4);
        u_slope5=(SourceX-x5)/(SourceY-y5);
        u_slope6=(SourceX-x6)/(SourceY-y6);
        u_slope7=(SourceX-x7)/(SourceY-y7);
        u_slope8=(SourceX-x8)/(SourceY-y8);
        slope_min=fmin(u_slope1,fmin(u_slope2,fmin(u_slope3,fmin(u_slope4,fmin(u_slope5,fmin(u_slope6,fmin(u_slope7,u_slope8)))))));
        slope_max=fmax(u_slope1,fmax(u_slope2,fmax(u_slope3,fmax(u_slope4,fmax(u_slope5,fmax(u_slope6,fmax(u_slope7,u_slope8)))))));
        coord_u1=(slope_min*DetectorY)+(SourceX-slope_min*SourceY);
        coord_u2=(slope_max*DetectorY)+(SourceX-slope_max*SourceY);
        index_u1=floor((coord_u1-u_plane[0])*1.0/du);
        index_u2=floor((coord_u2-u_plane[0])*1.0/du);
        s_index_u=min(index_u1,index_u2);
        e_index_u=max(index_u1,index_u2);
        v_slope1=(SourceZ-z1)/(SourceY-y1);
        v_slope2=(SourceZ-z2)/(SourceY-y2);
        v_slope3=(SourceZ-z3)/(SourceY-y3);
        v_slope4=(SourceZ-z4)/(SourceY-y4);
        v_slope5=(SourceZ-z5)/(SourceY-y5);
        v_slope6=(SourceZ-z6)/(SourceY-y6);
        v_slope7=(SourceZ-z7)/(SourceY-y7);
        v_slope8=(SourceZ-z8)/(SourceY-y8);
        slope_min=fmin(v_slope1,fmin(v_slope2,fmin(v_slope3,fmin(v_slope4,fmin(v_slope5,fmin(v_slope6,fmin(v_slope7,v_slope8)))))));
        slope_max=fmax(v_slope1,fmax(v_slope2,fmax(v_slope3,fmax(v_slope4,fmax(v_slope5,fmax(v_slope6,fmax(v_slope7,v_slope8)))))));
        coord_v1=(slope_min*DetectorY)+(SourceZ-slope_min*SourceY);
        coord_v2=(slope_max*DetectorY)+(SourceZ-slope_max*SourceY);
        index_v1=floor((coord_v1-v_plane[0])*1.0/dv);
        index_v2=floor((coord_v2-v_plane[0])*1.0/dv);
        s_index_v=min(index_v1,index_v2);
        e_index_v=max(index_v1,index_v2);
        InterpWeight=(R*R)/((R-yc)*(R-yc));
        for(k=s_index_v;k<=e_index_v;k++){
            if(k>=0 && k<=nv-1){
                if(s_index_v==e_index_v){
                    weight1=1.0;
                }else if(k==s_index_v){
                    weight1=(fmax(coord_v1,coord_v2)-v_plane[k+1])/fabs(coord_v1-coord_v2);
                }else if(k==e_index_v){
                    weight1=(v_plane[k]-fmin(coord_v1,coord_v2))/fabs(coord_v1-coord_v2);
                }else{
                    weight1=fabs(dv)/fabs(coord_v1-coord_v2);
                }
                for(l=s_index_u;l<=e_index_u;l++){
                    if(l>=0 && l<=nu-1){
                        if(s_index_u==e_index_u){
                            weight2=1.0;
                        }else if(l==s_index_u){
                            weight2=(u_plane[l+1]-fmin(coord_u1,coord_u2))/fabs(coord_u1-coord_u2);
                        }else if(l==e_index_u){
                            weight2=(fmax(coord_u1,coord_u2)-u_plane[l])/fabs(coord_u1-coord_u2);
                        }else{
                            weight2=fabs(du)/fabs(coord_u1-coord_u2);
                        }
                        atomicAdd(&Dest[tid],Src[k*nu+l]*weight1*weight2*InterpWeight);
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

class Backward:
    def __init__(self, filename, params):
        self.params = {'SourceInit': [0, 0, 0], 'DetectorInit': [0, 0, 0], 'StartAngle': 0,
                      'EndAngle': 0, 'NumberOfDetectorPixels': [0, 0], 'DetectorPixelSize': [0, 0],
                      'NumberOfViews': 0, 'ImagePixelSpacing': [0, 0, 0], 'NumberOfImage': [0, 0, 0],
                      'PhantomCenter': [0, 0, 0], 'Origin':[0, 0, 0], 'Method':'Distance', 'cutoff':0,
                      'FilterType':'hann', 'GPU':0}
        self.params = params
        start_time = time.time()
        self.proj = self.LoadProj(filename, [self.params['NumberOfViews'], self.params['NumberOfDetectorPixels'][1], self.params['NumberOfDetectorPixels'][0]], dtype=np.float32)
        # print('File load: ' + str(time.time() - start_time))
    @staticmethod
    def LoadProj(filename, image_size, dtype=np.float32):
        proj = np.fromfile(filename, dtype=dtype).reshape(image_size)
        return proj
    @staticmethod
    def LoadFile(filename, image_size, dtype=np.float32):
        image = np.fromfile(filename, dtype=dtype).reshape(image_size)
        return image

    def SaveProj(self, filename):
        self.proj.tofile(filename, sep='', format='')
    def SaveRecon(self, filename):
        self.recon.tofile(filename, sep='', format='')
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
        x = np.arange(0, N) - (N - 1) / 2
        h = np.zeros(len(x))
        h[np.where(x == 0)] = 1 / (8 * pixel_size ** 2)
        odds = np.where(x % 2 == 1)
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
            filter = filter * sin(w / (2 * cutoff)) / (w / (2 * cutoff))
            filter[zero] = tmp * sin(w[zero] / (2 * cutoff))
        elif FilterType == 'cosine':
            filter = filter * cos(w / (2 * cutoff))
        elif FilterType == 'hamming':
            filter = filter * (0.54 + 0.46 * (cos(w / cutoff)))
        elif FilterType == 'hann':
            filter = filter * (0.5 + 0.5 * cos(w / cutoff))

        filter[np.where(abs(w) > pi * cutoff / (2 * pixel_size))] = 0
        return filter

    def backward(self):
        start_time = time.time()
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
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2))
        # Calculates detector center
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = angle[0:-1]
        dtheta = angle[1] - angle[0]
        deltaS = du * SAD / SDD
        Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx + 1)) * dx
        Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny + 1)) * dy
        Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz + 1)) * dz
        Xpixel = Xplane[0:-1]
        Ypixel = Yplane[0:-1]
        Zpixel = Zplane[0:-1]
        ki = (np.arange(0, nu + 1) - nu / 2.0) * du
        p = (np.arange(0, nv + 1) - nv / 2.0) * dv
        alpha = 0
        beta = 0
        gamma = 0
        eu = [cos(gamma) * cos(alpha), sin(alpha), sin(gamma)]
        ev = [cos(gamma) * -sin(alpha), cos(gamma) * cos(alpha), sin(gamma)]
        ew = [0, 0, 1]
        print('Variable initialization: ' + str(time.time() - start_time))
        Source = [-SAD * sin(angle[0]), SAD * cos(angle[0]), 0]  # z-direction rotation
        Detector = [(SDD - SAD) * sin(angle[0]), -(SDD - SAD) * cos(angle[0]), 0]
        DetectorLength = np.array([np.arange(floor(-nu / 2), floor(nu / 2) + 1) * du, np.arange(floor(-nv / 2), floor(nv / 2) + 1) * dv])
        DetectorVectors = [eu, ev, ew]
        DetectorIndex = self.DetectorConstruction(Detector, DetectorLength, DetectorVectors, angle[0])
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        # plt.plot(ki)
        # plt.show()
        for i in range(nViews):
        # for i in range(12, 13):
            print(i)
            start_time = time.time()
            # print('Detector initialization: ' + str(time.time() - start_time))
            if(self.params['Method'] == 'Distance'):
                start_time = time.time()
                recon += self.distance_backproj(self.proj[i, :, :], DetectorIndex, angle[i], Xpixel, Ypixel, Zpixel, ki, p) * dtheta
                # print('Total backprojection: ' + str(time.time() - start_time))
#                 plt.imshow(recon[127, :, :], cmap='gray')
#                 plt.show()
            elif(self.params['Method'] == 'Ray'):
                recon += self.ray(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)
            # print('time taken: ' + str(time.time() - start_time) + '\n')
        self.recon = recon

    @staticmethod
    def filter_proj(proj, ki, p, params):
        [du, dv] = params['DetectorPixelSize']
        [nu, nv] = params['NumberOfDetectorPixels']
        ZeroPaddedLength = int(2 ** (ceil(log2(2 * (nu - 1)))))
        R = sqrt(np.sum((np.array(params['SourceInit']) - np.array(params['PhantomCenter'])) ** 2))
        D = sqrt(np.sum((np.array(params['DetectorInit']) - np.array(params['PhantomCenter'])) ** 2))
        # print(ki.shape, p.shape)
        [kk, pp] = np.meshgrid(ki[0:-1] * R / (R + D), p[0:-1] * R / (R + D))
        weight = R / (sqrt(R ** 2 + kk ** 2 + pp ** 2))

        deltaS = du * R / (R + D)
        filter = Backward.Filter(
            ZeroPaddedLength + 1, du * R / (D + R), params['FilterType'], params['cutoff'])
        weightd_proj = weight * proj
        Q = np.zeros(weightd_proj.shape, dtype=np.float32)
        for k in range(nv):
            tmp = real(ifft(
                ifftshift(filter * fftshift(fft(weightd_proj[k, :], ZeroPaddedLength)))))
            Q[k, :] = tmp[0:nu] * deltaS

        return Q
    def distance_backproj(self, proj, DetectorIndex, angle, Xpixel, Ypixel, Zpixel, ki, p):
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        Source = self.params['SourceInit']
        Detector = self.params['DetectorInit']
        R = sqrt(np.sum((np.array(Source) - np.array(self.params['PhantomCenter'])) ** 2))
        rotation_vector = [0, 0, 1]
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        Q = self.filter_proj(proj, ki, p, self.params)
#         [yy, xx] = np.meshgrid(Ypixel, Xpixel, indexing='ij')
#         ReconX_c1=(xx+dx/2)*cos(angle)+(yy+dy/2)*sin(angle)
#         ReconY_c1=-(xx+dx/2)*sin(angle)+(yy+dy/2)*cos(angle)
#         ReconX_c2=(xx-dx/2)*cos(angle)+(yy-dy/2)*sin(angle)
#         ReconY_c2=-(xx-dx/2)*sin(angle)+(yy-dy/2)*cos(angle)
#         ReconX_c3=(xx+dx/2)*cos(angle)+(yy-dy/2)*sin(angle)
#         ReconY_c3=-(xx+dx/2)*sin(angle)+(yy-dy/2)*cos(angle)
#         ReconX_c4=(xx-dx/2)*cos(angle)+(yy+dy/2)*sin(angle)
#         ReconY_c4=-(xx-dx/2)*sin(angle)+(yy+dy/2)*cos(angle)
        # plt.imshow(Q,cmap='gray')
        # plt.show()
        if self.params['GPU']:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            distance_backproj_about_z_gpu = mod.get_function("distance_backproj_about_z")
            distance_backproj_arb = mod.get_function("distance_backproj_arb")
    #             distance_proj_on_y_gpu = mod.get_function("distance_backproject_on_y2")
    #             distance_proj_on_x_gpu = mod.get_function("distance_project_on_x2")
    #             distance_proj_on_z_gpu = mod.get_function("distance_project_on_z2")
            Q = Q.flatten().astype(np.float32)
            dest = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
#             [zz, yy, xx] = np.meshgrid(Zpixel, Ypixel, Xpixel, indexing='ij')
            x_pixel_gpu = pycuda.gpuarray.to_gpu(Xpixel.astype(np.float32))
            y_pixel_gpu = pycuda.gpuarray.to_gpu(Ypixel.astype(np.float32))
            z_pixel_gpu = pycuda.gpuarray.to_gpu(Zpixel.astype(np.float32))
            u_plane_gpu = pycuda.gpuarray.to_gpu(ki.astype(np.float32))
            v_plane_gpu = pycuda.gpuarray.to_gpu(p.astype(np.float32))

        if(rotation_vector == [0, 0, 1]):
            start_time = time.time()
            if(self.params['GPU']):
                TotalSize = nx * ny * nz
                if(TotalSize < MAX_THREAD_PER_BLOCK):
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
                        if(GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Backward._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()

                recon_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1], Detector[2], angle, 0.0, R]).astype(np.float32)
                recon_param_gpu = pycuda.gpuarray.to_gpu(recon_param)

                #distance_backproj_about_z_gpu(dest, drv.In(Q), x_pixel_gpu, y_pixel_gpu,
                #                 z_pixel_gpu, u_plane_gpu, v_plane_gpu, recon_param_gpu,
                #                      block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                distance_backproj_arb(dest, drv.In(Q), x_pixel_gpu, y_pixel_gpu,
                                 z_pixel_gpu, u_plane_gpu, v_plane_gpu, recon_param_gpu,
                                      block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del u_plane_gpu, v_plane_gpu, x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, recon_param_gpu
                recon = dest.get().reshape([nz, ny, nx]).astype(np.float32)
                del dest
            else:
                recon = self._distance_backproj_arb(Q, Xpixel, Ypixel, Zpixel, ki, p, angle, 0.0, self.params)  # * intersection_length
        elif(rotation_vector == [0, 1, 0]):
            pass
        elif(rotation_vector == [1, 0, 0]):
            pass

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
        R = sqrt(np.sum((np.array(Source) - np.array(params['PhantomCenter'])) ** 2))
        recon_pixelsX = Xpixel
        recon_pixelsY = Ypixel
        recon_pixelsZ = Zpixel
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        f_angle = lambda x, y: atan(x / y) if y != 0 else atan(0) if x == 0 else -pi / 2 if x < 0 else pi / 2
        fx = lambda x, y, z:x * cos(angle2) * cos(angle1) + y * cos(angle2) * sin(angle1) - z * sin(angle2) * cos(angle1) * sin(f_angle(x, y)) - z * sin(angle2) * sin(angle1) * cos(f_angle(x, y))
        fy = lambda x, y, z:y * cos(angle2) * cos(angle1) - x * cos(angle2) * sin(angle1) - z * sin(angle2) * cos(angle1) * cos(f_angle(x, y)) + z * sin(angle2) * sin(angle1) * sin(f_angle(x, y))
        fz = lambda x, y, z:z * cos(angle2) + sqrt(x ** 2 + y ** 2) * sin(angle2)
        for i in range(127, 128):
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
                        if(l < 0 or l > nu):
                            continue
                        if(s_index_v == e_index_v):
                            weight1 = 1.0
                        elif(l == s_index_v):
                            weight1 = (max(coord_v1, coord_v2) - Vplane[l + 1]) / abs(coord_v1 - coord_v2)
                        elif(l == e_index_v):
                            weight1 = (Vplane[l] - min(coord_v1, coord_v2)) / abs(coord_v1 - coord_v2)
                        else:
                            weight1 = abs(dv) / abs(coord_v1 - coord_v2)
                        for m in range(s_index_u, e_index_u + 1):
                            if(m < 0 or m > nv):
                                continue
                            if(s_index_u == e_index_u):
                                weight2 = 1.0
                            elif(m == s_index_u):
                                weight2 = (Uplane[k + 1] - min(coord_u1, coord_u2)) / abs(coord_u1 - coord_u2)
                            elif(m == e_index_u):
                                weight2 = (max(coord_u1, coord_u2) - Uplane[k]) / abs(coord_u1 - coord_u2)
                            else:
                                weight2 = abs(du) / abs(coord_u1 - coord_u2)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2 * (R ** 2) / (R - yc) ** 2
            plt.imshow(recon[i, :, :], cmap='gray')
            plt.show()
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
        for i in range(127, 128):
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
                        if(s_index_v == e_index_v):
                            weight1 = 1.0
                        elif(l == s_index_v):
                            weight1 = (max(coord_v1, coord_v2) - Vplane[l + 1]) / abs(coord_v1 - coord_v2)
                        elif(l == e_index_v):
                            weight1 = (Vplane[l] - min(coord_v1, coord_v2)) / abs(coord_v1 - coord_v2)
                        else:
                            weight1 = abs(dv) / abs(coord_v1 - coord_v2)
                        for m in range(s_index_u, e_index_u + 1):
                            if(s_index_u == e_index_u):
                                weight2 = 1.0
                            elif(m == s_index_u):
                                weight2 = (Uplane[k + 1] - min(coord_u1, coord_u2)) / abs(coord_u1 - coord_u2)
                            elif(m == e_index_u):
                                weight2 = (max(coord_u1, coord_u2) - Uplane[k]) / abs(coord_u1 - coord_u2)
                            else:
                                weight2 = abs(du) / abs(coord_u1 - coord_u2)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2 * (R ** 2) / (R - yc) ** 2
        return recon
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
                        weight1 = abs(dy) / abs(CoordY1[i, j] - CoordY2[i, j])
                    # if(abs(weight1) - 0 < tol_min):
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
                        # if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        # print(weight1,weight2)
                        assert(weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[l][k][ix]
                proj[i, j] = p_value
        return proj

    @staticmethod
    def _distance_project_on_z(image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane, image_x1, image_X2, image_y1, image_y2, dx, dy, iz):
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
                    if(k < 0 or k > image.shape[0] - 1):
                        continue
                    if(s_index_x == e_index_x):
                        weight1 = 1
                    elif(k == s_index_x):
                        weight1 = (Xplane[k + 1] - max(CoordX1[i, j], CoordX2[i, j])) / abs(CoordX1[i, j] - CoordX2[i, j])
                    elif(k == e_index_x):
                        weight1 = (min(CoordY1[i, j], CoordY2[i, j]) - Xplane[k ]) / abs(CoordX1[i, j] - CoordX2[i, j])
                    else:
                        weight1 = abs(dx) / abs(CoordX1[i, j] - CoordX2[i, j])
                    # if(abs(weight1) - 0 < tol_min):
                    #    weight1 = 0
                    for l in range(int(s_index_y), int(e_index_y) + 1):
                        if(l < 0 or l > image.shape[1] - 1):
                            continue
                        if(s_index_z == e_index_z):
                            weight2 = 1
                        elif(l == s_index_y):
                            weight2 = (max(CoordY1[i, j], CoordY2[i, j]) - Yplane[l + 1]) / abs(CoordY1[i, j] - CoordY2[i, j])
                        elif(l == e_index_y):
                            weight2 = (Yplane[l] - min(CoordY1[i, j], CoordY2[i, j])) / abs(CoordY1[i, j] - CoordY2[i, j])
                        else:
                            weight2 = abs(dy) / abs(CoordY1[i, j] - CoordY2[i, j])
                        # print(s_index_z,e_index_z,Zplane[l+1],Zplane[l],CoordZ1[i,j],CoordZ2[i,j])
                        # if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        # print(weight1,weight2)
                        assert(weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
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
    filename = 'proj_distance.dat'
    params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
                      'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.5, 0.5],
                      'NumberOfViews': 720, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
                      'PhantomCenter': [0, 0, 0], 'Origin':[0, 0, 0], 'Method':'Distance', 'FilterType':'hann' , 'cutoff': 1, 'GPU':1}
    F = Backward(filename, params)
    F.backward()
    F.recon.tofile('recon_distance.dat', sep='', format='')
    end_time = time.time()
    # plt.imshow(F.recon[0, :, :], cmap='gray')
    # plt.show()

if __name__ == '__main__':
    main()
