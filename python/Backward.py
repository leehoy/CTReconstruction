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
        int ix=(int) floor(tid*1.0 /N);
        int pix_num=tid-(N*ix); 
        float coord_x1,coord_x2,coord_y1,coord_y2;
        int index_x1,index_x2,index_y1,index_y2;
        coord_x1=slope_x1[pix_num]*(Zplane[iz]+dz/2)+intercept_x1[pix_num];
        coord_x2=slope_x2[pix_num]*(Zplane[iz]+dz/2)+intercept_x2[pix_num];
        coord_z1=slope_y1[pix_num]*(Zplane[iz]+dz/2)+intercept_y1[pix_num];
        coord_z2=slope_y2[pix_num]*(Zplane[iz]+dz/2)+intercept_y2[pix_num];
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
                        weight1=(Xplane[k+1]-fmin(coord_x1,coord_x2)/fabs(coord_x1-coord_x2);
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
    float dx=param[0];
    float dy=param[1];
    float dz=param[2];
    float du=params[8],dv=params[9];
    int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7];
    int k=0,l=0,N=nx*ny*nz,ix=0,iy=0,iz=0;
    
    if(tid<N){
        iz=(int)(N*1.0)/(nx*ny);
        iy=(int)(N-iz*nx*ny)/(nx*1.0);
        ix=(int)(N-iz*nx*ny-iy*nx);
        
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
        deltaS = du * SAD / SDD
        proj = np.zeros([nViews, nv, nu], dtype=np.float32)
        gamma = (range(0, nu - 1) - (nu - 1) / 2) * deltaS
        Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx + 1)) * dx
        Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny + 1)) * dy
        Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz + 1)) * dz
        Xplane = Xplane - dx / 2
        Yplane = Yplane - dy / 2
        Zplane = Zplane - dz / 2
        ki = (np.arange(0, nu + 1) - nu / 2.0) * du
        p = (np.arange(0, nv + 1) - nv / 2.0) * dv
        cutoff = nu
        FilterType = 'hann'
        filter = ConeBeam.Filter(
            ZeroPaddedLength + 1, du * R / (D + R), FilterType, cutoff)
#         ki = (ki * R) / (R + D)
#         p = (p * R) / (R + D)
        [kk, pp] = np.meshgrid(ki[0:-1] * R / (R + D), p[0:-1] * R / (R + D))
        weight = R / (sqrt(R ** 2 + kk ** 2 + pp ** 2))
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
        DetectorIndex = self.DetectorConstruction(Detector, DetectorLength, DetectorVectors, angle[i])
        for i in range(nViews):
        # for i in range(12, 13):
            print(i)
            start_time = time.time()
            # print('Detector initialization: ' + str(time.time() - start_time))
            if(self.params['Method'] == 'Distance'):
                start_time = time.time()
                recon += self.distance_backproj(self.proj[i, :, :], Xplane, Yplane, Zplane, angle[i], Source, Detector, ki, p)
                print('Total backprojection: ' + str(time.time() - start_time))
            elif(self.params['Method'] == 'Ray'):
                recon += self.ray(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)
            print('time taken: ' + str(time.time() - start_time) + '\n')
        self.proj = proj

    def distance_backproj(self, proj, DetectorIndex, Source, Detector, angle, Xplane, Yplane, Zplane):
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        if self.params['GPU']:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            distance_backproj_about_z_gpu = mod.get_function("distance_backproj_about_z")
    #             distance_proj_on_y_gpu = mod.get_function("distance_backproject_on_y2")
    #             distance_proj_on_x_gpu = mod.get_function("distance_project_on_x2")
    #             distance_proj_on_z_gpu = mod.get_function("distance_project_on_z2")
            image = self.image.flatten().astype(np.float32)
            dest = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
            x_plane_gpu = pycuda.gpuarray.to_gpu(Xplane.astype(np.float32))
            y_plane_gpu = pycuda.gpuarray.to_gpu(Yplane.astype(np.float32))
            z_plane_gpu = pycuda.gpuarray.to_gpu(Zplane.astype(np.float32))
        WeightedProjection = weight * proj
        Q = np.zeros(WeightedProjection.shape)
        for k in range(nv):
            tmp = real(ifft(
                ifftshift(filter * fftshift(fft(WeightedProjection[k, :], ZeroPaddedLength)))))
            Q[k, :] = tmp[0:nu]
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
                recon_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                recon_param_gpu = pycuda.gpuarray.to_gpu(recon_param)
                
                distance_backproj_about_z_gpu(dest, drv.In(image), x_plane_gpu, y_plane_gpu,
                                  z_plane_gpu, u_plane_gpu, v_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del u_plane_gpu, v_plane_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                recon = dest.get().reshape([nz, ny, nx]).astype(np.float32)
                del dest
            else:
                recon += self._distance_backproj_about_z(self.proj, Xplane, Yplane, Zplane, Uplane, Vplane, param) * intersection_length
        elif(rotation_vector == [0, 1, 0]):
            pass
        elif(rotation_vector == [1, 0, 0]):
            pass

        return recon
    @staticmethod
    def _distacne_backproj_about_z(proj, Xplane, Yplane, Zplane, Source, Detector, Uplane, Vplane, param):
        tol_min = 1e-6
        nx = params['nx']
        ny = params['ny']
        nz = parasm['nz']
        dx = params['dx']
        dy = params['dy']
        dz = params['dz']
        du = params['du']
        dv = params['dv']
        SAD = params['SAD']
        SDD = parasm['SDD']
        angle = params['angle']
        bp = np.zeros([nz, ny, nx], dtype=params.dtype)
        recon_pixelsX = Xplane[0:-1] + dx / 2
        recon_pixelsY = Yplane[0:-1] + dy / 2
        recon_pixelsZ = Zplane[0:-1] + dz / 2
        [reconY, reconX] = np.meshgrid(recon_pixelsY, recon_pixlesZ)
        reconX_c1 = (reconX + dx / 2) * cos(angle) + (reconY + dy / 2) * sin(angle)
        reconX_c2 = (reconX - dx / 2) * cos(angle) + (reconY - dy / 2) * sin(angle)
        reconX_c3 = (reconX + dx / 2) * cos(angle) + (reconY - dy / 2) * sin(angle)
        reconX_c4 = (reconX - dx / 2) * cos(angle) + (reconY + dy / 2) * sin(angle)
        
        reconY_c1 = -(reconX + dx / 2) * sin(angle) + (reconY + dy / 2) * cos(angle)
        reconY_c2 = -(reconX - dx / 2) * sin(angle) + (reconY - dy / 2) * cos(angle)
        reconY_c3 = -(reconX + dx / 2) * sin(angle) + (reconY - dy / 2) * cos(angle)
        reconY_c4 = -(reconX - dx / 2) * sin(angle) + (reconY + dy / 2) * cos(angle)
        SlopeU_c1 = (Source[0] - reconX_c1) / (Source[1] - reconY_c1)
        SlopeU_c2 = (Source[0] - reconX_c2) / (Source[1] - reconY_c2)
        SlopeU_c3 = (Source[0] - reconX_c3) / (Source[1] - reconY_c3)
        SlopeU_c4 = (Source[0] - reconX_c4) / (Source[1] - reconY_c4)
        [reconZ,reconY]=np.meshgrid
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    x1 = (recon_pixelsX[k] + dx / 2) * cos(angle) + (recon_pixelsY[j] + dy / 2) * sin(angle)
                    y1 = -(recon_pixelsX[k] + dx / 2) * sin(angle) + (recon_pixelsY[j] + dy / 2) * cos(angle)
                    slope1 = (x1 / (-y1 + SAD)) 
                    x2 = (recon_pixelsX[k] - dx / 2) * cos(angle) + (recon_pixelsY[j] - dy / 2) * sin(angle)
                    y2 = -(recon_pixelsX[k] - dx / 2) * sin(angle) + (recon_pixelsY[j] - dy / 2) * cos(angle)
                    slope2 = (x2 / (-y2 + SAD)) 
                    x3 = (recon_pixelsX[k] + dx / 2) * cos(angle) + (recon_pixelsY[j] - dy / 2) * sin(angle)
                    y3 = -(recon_pixelsX[k] + dx / 2) * sin(angle) + (recon_pixelsY[j] - dy / 2) * cos(angle)
                    slope3 = (x3 / (-y3 + SAD)) 
                    x4 = (recon_pixelsX[k] - dx / 2) * cos(angle) + (recon_pixelsY[j] + dy / 2) * sin(angle)
                    y4 = -(recon_pixelsX[k] - dx / 2) * sin(angle) + (recon_pixelsY[j] + dy / 2) * cos(angle)
                    slope4 = (x4 / (-y4 + SAD))
                    slopes_u = [x1 / (-y1 + SAD), x2 / (-y2 + SAD), x3 / (-y3 + SAD), x4 / (-y4 + SAD)] 
                    slope_l = min(slopes_u)
                    slope_r = max(slopes_u)
                    u_l = slope_l * SDD / du + nu / 2
                    u_r = slope_r * SDD / du + nu / 2
                    z1 = recon_pixelsZ[i] + sqrt((dy / 2) ** 2 + (dz / 2) ** 2)
                    z2 = recon_pixelsZ[i] - sqrt((dy / 2) ** 2 + (dz / 2) ** 2)
                    slopes_v = [z1 / (-y1 + SAD), z1 / (-y2 + SAD), z1 / (-y3 + SAD), z1 / (-y4 + SAD), z2 / (-y1 + SAD), z2 / (-y2 + SAD), z3 / (-y3 + SAD), z4 / (-y4 + SAD)]
                    slope_t = min(slopes_v)
                    slope_b = max(slopes_v)
                    v_t = slope_t * SDD / dv + nv / 2
                    v_b = slope_b * SDD / dv + nv / 2
                    for l in range(floor(v_min), floor(v_max) + 1):
                        if(ceil(v_min) == floor(v_max)):
                            weight1 = 1.0
                        elif(l == floor(v_min)):
                            weight1 = (ceil(v_min) - v_min) / abs(v_max - v_min)
                        elif(l == floor(v_max)):
                            weight1 = (v_max - floor(v_max)) / abs(v_max - v_min)
                        else:
                            weight1 = abs(dv) / abs(v_max - v_min)
                        for m in range(floor(u_min), floor(u_max) + 1):
                            if(ceil(u_min) == floor(u_max)):
                                weight2 = 1.0
                            elif(m == floor(u_min)):
                                weight2 = (ceil(u_min) - u_min) / abs(u_max - u_min)
                            elif(m == floor(u_max)):
                                weight2 = (u_max - floor(u_max)) / abs(u_max - u_min)
                            else:
                                weight2 = abs(du) / abs(u_max - u_min)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2
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
    filename = ''
    params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
                      'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.5, 0.5],
                      'NumberOfViews': 720, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
                      'PhantomCenter': [0, 0, 0], 'Origin':[0, 0, 0], 'Method':'Distance', 'GPU':1}
    F = Backward(filename, params)
    F.backward()
    F.recon.tofile('proj_distance.dat', sep='', format='')
    end_time = time.time()
    # plt.imshow(F.recon[0, :, :], cmap='gray')
    # plt.show()

if __name__ == '__main__':
    main()
