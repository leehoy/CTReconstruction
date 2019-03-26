import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


def DefineGPUFuns():
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
            w1=1.0f;
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
                w1=1.0f;
            if(fabs(w3-0)<0.0001 && fabs(w4-0)<0.0001)
                w3=1.0f;
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
                            weight1=1.0f;
                        }else if(k==s_index_x){
                            weight1=(Xplane[k+1]-fminf(CoordX1[tid],CoordX2[tid]))/fabsf(CoordX1[tid]-CoordX2[tid]);
                        }else if(k==e_index_x){
                            weight1=(fmaxf(CoordX1[tid],CoordX2[tid])-Xplane[k])/fabsf(CoordX1[tid]-CoordX2[tid]);
                        }else{
                            weight1=fabsf(dx)/fabsf(CoordX1[tid]-CoordX2[tid]);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_z;l<=e_index_z;l++){
                            if(l>=0 && l<= nz-1){
                                if(s_index_z==e_index_z){
                                    weight2=1.0f;
                                }else if(l==s_index_z){
                                    weight2=(fmaxf(CoordZ1[tid],CoordZ2[tid])-Zplane[l+1])/fabsf(CoordZ1[tid]-CoordZ2[tid]);
                                }else if(l==e_index_z){
                                    weight2=(Zplane[l]-fminf(CoordZ1[tid],CoordZ2[tid]))/fabsf(CoordZ1[tid]-CoordZ2[tid]);
                                }else{
                                    weight2=fabsf(dz)/fabsf(CoordZ1[tid]-CoordZ2[tid]);
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
                            weight1=1.0f;
                        }else if(k==s_index_y){
                            weight1=(fmaxf(CoordY1[tid],CoordY2[tid])-Yplane[k+1])/fabsf(CoordY1[tid]-CoordY2[tid]);
                        }else if(k==e_index_y){
                            weight1=(Yplane[k]-fmin(CoordY1[tid],CoordY2[tid]))/fabsf(CoordY1[tid]-CoordY2[tid]);
                        }else{
                            weight1=fabsf(dy)/fabsf(CoordY1[tid]-CoordY2[tid]);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_z;l<=e_index_z;l++){
                            if(l>=0 && l<=nz-1){
                                if(s_index_z==e_index_z){
                                    weight2=1.0f;
                                }else if(l==s_index_z){
                                    weight2=(fmaxf(CoordZ1[tid],CoordZ2[tid])-Zplane[l+1])/fabsf(CoordZ1[tid]-CoordZ2[tid]);
                                }else if(l==e_index_z){
                                    weight2=(Zplane[l]-fminf(CoordZ1[tid],CoordZ2[tid]))/fabsf(CoordZ1[tid]-CoordZ2[tid]);
                                }else{
                                    weight2=fabsf(dz)/fabsf(CoordZ1[tid]-CoordZ2[tid]);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
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
                            weight1=1.0f;
                        }else if(k==s_index_x){
                            weight1=(Xplane[k+1]-fminf(coord_x1,coord_x2))/fabsf(coord_x1-coord_x2);
                        }else if(k==e_index_x){
                            weight1=(fmaxf(coord_x1,coord_x2)-Xplane[k])/fabsf(coord_x1-coord_x2);
                        }else{
                            weight1=fabsf(dx)/fabsf(coord_x1-coord_x2);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_z;l<=e_index_z;l++){
                            if(l>=0 && l<= nz-1){
                                if(s_index_z==e_index_z){
                                    weight2=1.0f;
                                }else if(l==s_index_z){
                                    weight2=(fmaxf(coord_z1,coord_z2)-Zplane[l+1])/fabsf(coord_z1-coord_z2);
                                }else if(l==e_index_z){
                                    weight2=(Zplane[l]-fmin(coord_z1,coord_z2))/fabsf(coord_z1-coord_z2);
                                }else{
                                    weight2=fabsf(dz)/fabsf(coord_z1-coord_z2);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
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
                            weight1=1.0f;
                        }else if(k==s_index_y){
                            weight1=(fmaxf(coord_y1,coord_y2)-Yplane[k+1])/fabsf(coord_y1-coord_y2);
                        }else if(k==e_index_y){
                            weight1=(Yplane[k]-fminf(coord_y1,coord_y2))/fabsf(coord_y1-coord_y2);
                        }else{
                            weight1=fabsf(dy)/fabsf(coord_y1-coord_y2);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_z;l<=e_index_z;l++){
                            if(l>=0 && l<=nz-1){
                                if(s_index_z==e_index_z){
                                    weight2=1.0f;
                                }else if(l==s_index_z){
                                    weight2=(fmaxf(coord_z1,coord_z2)-Zplane[l+1])/fabsf(coord_z1-coord_z2);
                                }else if(l==e_index_z){
                                    weight2=(Zplane[l]-fminf(coord_z1,coord_z2))/fabsf(coord_z1-coord_z2);
                                }else{
                                    weight2=fabsf(dz)/fabsf(coord_z1-coord_z2);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
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
                            weight1=1.0f;
                        }else if(k==s_index_x){
                            weight1=(Xplane[k+1]-fminf(coord_x1,coord_x2))/fabsf(coord_x1-coord_x2);
                        }else if(k==e_index_x){
                            weight1=(fmaxf(coord_x1,coord_x2)-Xplane[k])/fabsf(coord_x1-coord_x2);
                        }else{
                            weight1=fabsf(dx)/fabsf(coord_x1-coord_x2);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_y;l<=e_index_y;l++){
                            if(l>=0 && l<=ny-1){
                                if(s_index_y==e_index_y){
                                    weight2=1.0f;
                                }else if(l==s_index_y){
                                    weight2=(fmaxf(coord_y1,coord_y2)-Yplane[l+1])/fabsf(coord_y1-coord_y2);
                                }else if(l==e_index_y){
                                    weight2=(Yplane[l]-fminf(coord_y1,coord_y2))/fabsf(coord_y1-coord_y2);
                                }else{
                                    weight2=fabsf(dy)/fabsf(coord_y1-coord_y2);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
                                }
                                atomicAdd(&Dest[pix_num],Src[(iz*nx*ny)+l*nx+k]*weight1*weight2);
                            }
                        }
                    }
                }
            }
    }
    
    __global__ void distance_project_on_y3(float* Dest,float* Src,float* slope_x1,float* slope_x2,float* slope_z1,float* slope_z2,float* intercept_x1,float* intercept_x2,float* intercept_z1,float* intercept_z2,float* Xplane,float* Yplane,float* Zplane,float* intersection,float* param){
            int x=blockDim.x*blockIdx.x+threadIdx.x;
            int y=blockDim.y*blockIdx.y+threadIdx.y;
            int tid=y*gridDim.x*blockDim.x+x;
            float dx=param[0];
            float dy=param[1];
            float dz=param[2];
            int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7],angle_ind=(int)param[8];
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
                            weight1=1.0f;
                        }else if(k==s_index_x){
                            weight1=(Xplane[k+1]-fminf(coord_x1,coord_x2))/fabsf(coord_x1-coord_x2);
                        }else if(k==e_index_x){
                            weight1=(fmaxf(coord_x1,coord_x2)-Xplane[k])/fabsf(coord_x1-coord_x2);
                        }else{
                            weight1=fabsf(dx)/fabsf(coord_x1-coord_x2);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_z;l<=e_index_z;l++){
                            if(l>=0 && l<= nz-1){
                                if(s_index_z==e_index_z){
                                    weight2=1.0f;
                                }else if(l==s_index_z){
                                    weight2=(fmaxf(coord_z1,coord_z2)-Zplane[l+1])/fabsf(coord_z1-coord_z2);
                                }else if(l==e_index_z){
                                    weight2=(Zplane[l]-fmin(coord_z1,coord_z2))/fabsf(coord_z1-coord_z2);
                                }else{
                                    weight2=fabsf(dz)/fabsf(coord_z1-coord_z2);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
                                }
                                atomicAdd(&Dest[pix_num+angle_ind*N],Src[(l*nx*ny)+iy*nx+k]*weight1*weight2*intersection[pix_num]);
                            }
                            
                            //syncthreads();
                        }
                    }
                }
            }
    }
    
    __global__ void distance_project_on_x3(float* Dest,float* Src,float* slope_y1,float* slope_y2,float* slope_z1,float* slope_z2,float* intercept_y1,float* intercept_y2,float* intercept_z1,float* intercept_z2,float* Xplane,float* Yplane,float* Zplane,float* intersection,float* param){
            int x=blockDim.x*blockIdx.x+threadIdx.x;
            int y=blockDim.y*blockIdx.y+threadIdx.y;
            int tid=y*gridDim.x*blockDim.x+x;
            float dx=param[0];
            float dy=param[1];
            float dz=param[2];
            int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7],angle_ind=(int)param[8];
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
                            weight1=1.0f;
                        }else if(k==s_index_y){
                            weight1=(fmaxf(coord_y1,coord_y2)-Yplane[k+1])/fabsf(coord_y1-coord_y2);
                        }else if(k==e_index_y){
                            weight1=(Yplane[k]-fminf(coord_y1,coord_y2))/fabsf(coord_y1-coord_y2);
                        }else{
                            weight1=fabsf(dy)/fabsf(coord_y1-coord_y2);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_z;l<=e_index_z;l++){
                            if(l>=0 && l<=nz-1){
                                if(s_index_z==e_index_z){
                                    weight2=1.0f;
                                }else if(l==s_index_z){
                                    weight2=(fmaxf(coord_z1,coord_z2)-Zplane[l+1])/fabsf(coord_z1-coord_z2);
                                }else if(l==e_index_z){
                                    weight2=(Zplane[l]-fminf(coord_z1,coord_z2))/fabsf(coord_z1-coord_z2);
                                }else{
                                    weight2=fabsf(dz)/fabsf(coord_z1-coord_z2);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
                                }
                                atomicAdd(&Dest[pix_num+angle_ind*N],Src[(l*nx*ny)+k*nx+ix]*weight1*weight2*intersection[pix_num]);
                            }
                        }
                    }
                }
            }
    }
    
    __global__ void distance_project_on_z3(float* Dest,float* Src,float* slope_x1,float* slope_x2,float* slope_y1,float* slope_y2,float* intercept_x1,float* intercept_x2,float* intercept_y1,float* intercept_y2,float* Xplane,float* Yplane,float* Zplane,float* intersection,float* param){
            int x=blockDim.x*blockIdx.x+threadIdx.x;
            int y=blockDim.y*blockIdx.y+threadIdx.y;
            int tid=y*gridDim.x*blockDim.x+x;
            float dx=param[0];
            float dy=param[1];
            float dz=param[2];
            int nx=(int)param[3],ny=(int)param[4],nz=(int)param[5],nu=(int)param[6],nv=(int)param[7],angle_ind=(int)param[8];
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
                            weight1=1.0f;
                        }else if(k==s_index_x){
                            weight1=(Xplane[k+1]-fminf(coord_x1,coord_x2))/fabsf(coord_x1-coord_x2);
                        }else if(k==e_index_x){
                            weight1=(fmaxf(coord_x1,coord_x2)-Xplane[k])/fabsf(coord_x1-coord_x2);
                        }else{
                            weight1=fabsf(dx)/fabsf(coord_x1-coord_x2);
                        }
                        if(fabs(weight1)<0.000001){
                            weight1=0.0f;
                        }
                        for(l=s_index_y;l<=e_index_y;l++){
                            if(l>=0 && l<=ny-1){
                                if(s_index_y==e_index_y){
                                    weight2=1.0f;
                                }else if(l==s_index_y){
                                    weight2=(fmaxf(coord_y1,coord_y2)-Yplane[l+1])/fabsf(coord_y1-coord_y2);
                                }else if(l==e_index_y){
                                    weight2=(Yplane[l]-fminf(coord_y1,coord_y2))/fabsf(coord_y1-coord_y2);
                                }else{
                                    weight2=fabsf(dy)/fabsf(coord_y1-coord_y2);
                                }
                                if(fabs(weight2)<0.000001){
                                    weight2=0.0f;
                                }
                                atomicAdd(&Dest[pix_num+angle_ind*N],Src[(iz*nx*ny)+l*nx+k]*weight1*weight2*intersection[pix_num]);
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
        x1=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y1=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z1=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x2=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y2=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z2=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x3=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y3=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z3=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x4=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y4=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z4=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x5=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y5=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z5=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x6=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y6=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z6=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x7=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y7=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z7=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x8=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y8=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z8=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);

        u_slope1=(SourceX-x1)/(SourceY-y1);
        u_slope2=(SourceX-x2)/(SourceY-y2);
        u_slope3=(SourceX-x3)/(SourceY-y3);
        u_slope4=(SourceX-x4)/(SourceY-y4);
        u_slope5=(SourceX-x5)/(SourceY-y5);
        u_slope6=(SourceX-x6)/(SourceY-y6);
        u_slope7=(SourceX-x7)/(SourceY-y7);
        u_slope8=(SourceX-x8)/(SourceY-y8);
        slope_min=fminf(u_slope1,fminf(u_slope2,fminf(u_slope3,fminf(u_slope4,fminf(u_slope5,fminf(u_slope6,fminf(u_slope7,u_slope8)))))));
        slope_max=fmaxf(u_slope1,fmaxf(u_slope2,fmaxf(u_slope3,fmaxf(u_slope4,fmaxf(u_slope5,fmaxf(u_slope6,fmaxf(u_slope7,u_slope8)))))));
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
        slope_min=fminf(v_slope1,fminf(v_slope2,fminf(v_slope3,fminf(v_slope4,fminf(v_slope5,fminf(v_slope6,fminf(v_slope7,v_slope8)))))));
        slope_max=fmaxf(v_slope1,fmaxf(v_slope2,fmaxf(v_slope3,fmaxf(v_slope4,fmaxf(v_slope5,fmaxf(v_slope6,fmaxf(v_slope7,v_slope8)))))));
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
                    weight1=1.0f;
                }else if(k==s_index_v){
                    weight1=(fmaxf(coord_v1,coord_v2)-v_plane[k+1])/fabsf(coord_v1-coord_v2);
                }else if(k==e_index_v){
                    weight1=(v_plane[k]-fminf(coord_v1,coord_v2))/fabsf(coord_v1-coord_v2);
                }else{
                    weight1=fabsf(dv)/fabsf(coord_v1-coord_v2);
                }
                for(l=s_index_u;l<=e_index_u;l++){
                    if(l>=0 && l<=nu-1){
                        if(s_index_u==e_index_u){
                            weight2=1.0f;
                        }else if(l==s_index_u){
                            weight2=(u_plane[l+1]-fminf(coord_u1,coord_u2))/fabsf(coord_u1-coord_u2);
                        }else if(l==e_index_u){
                            weight2=(fmaxf(coord_u1,coord_u2)-u_plane[l])/fabsf(coord_u1-coord_u2);
                        }else{
                            weight2=fabsf(du)/fabsf(coord_u1-coord_u2);
                        }
                        atomicAdd(&Dest[tid],Src[k*nu+l]*weight1*weight2*InterpWeight);
                        //__syncthreads();
                    }                    
                }
            }
        }
    }
}
__global__ void flat_distance_backproj_arb2(float* Dest, float* Src,float* x_plane,float* y_plane,float* z_plane,float* u_plane,float* v_plane,float* params){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int tid=y*gridDim.x*blockDim.x+x;
    float dx=params[0],dy=params[1],dz=params[2];
    int nx=(int)params[3],ny=(int)params[4],nz=(int)params[5],nu=(int)params[6],nv=(int)params[7];
    float du=params[8],dv=params[9];
    float SourceX=params[10],SourceY=params[11],SourceZ=params[12],DetectorY=params[14];
    float angle1=params[16], angle2=params[17];
    float R=params[18];
    int angle_ind=(int)params[19]; 
    float yc;
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
        x1=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y1=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z1=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x2=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y2=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z2=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x3=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y3=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z3=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x4=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y4=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z4=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x5=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y5=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z5=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x6=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y6=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z6=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x7=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y7=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z7=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x8=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y8=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z8=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);

        u_slope1=(SourceX-x1)/(SourceY-y1);
        u_slope2=(SourceX-x2)/(SourceY-y2);
        u_slope3=(SourceX-x3)/(SourceY-y3);
        u_slope4=(SourceX-x4)/(SourceY-y4);
        u_slope5=(SourceX-x5)/(SourceY-y5);
        u_slope6=(SourceX-x6)/(SourceY-y6);
        u_slope7=(SourceX-x7)/(SourceY-y7);
        u_slope8=(SourceX-x8)/(SourceY-y8);
        slope_min=fminf(u_slope1,fminf(u_slope2,fminf(u_slope3,fminf(u_slope4,fminf(u_slope5,fminf(u_slope6,fminf(u_slope7,u_slope8)))))));
        slope_max=fmaxf(u_slope1,fmaxf(u_slope2,fmaxf(u_slope3,fmaxf(u_slope4,fmaxf(u_slope5,fmaxf(u_slope6,fmaxf(u_slope7,u_slope8)))))));
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
        slope_min=fminf(v_slope1,fminf(v_slope2,fminf(v_slope3,fminf(v_slope4,fminf(v_slope5,fminf(v_slope6,fminf(v_slope7,v_slope8)))))));
        slope_max=fmaxf(v_slope1,fmaxf(v_slope2,fmaxf(v_slope3,fmaxf(v_slope4,fmaxf(v_slope5,fmaxf(v_slope6,fmaxf(v_slope7,v_slope8)))))));
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
                    weight1=1.0f;
                }else if(k==s_index_v){
                    weight1=(fmaxf(coord_v1,coord_v2)-v_plane[k+1])/fabsf(coord_v1-coord_v2);
                }else if(k==e_index_v){
                    weight1=(v_plane[k]-fminf(coord_v1,coord_v2))/fabsf(coord_v1-coord_v2);
                }else{
                    weight1=fabsf(dv)/fabsf(coord_v1-coord_v2);
                }
                for(l=s_index_u;l<=e_index_u;l++){
                    if(l>=0 && l<=nu-1){
                        if(s_index_u==e_index_u){
                            weight2=1.0f;
                        }else if(l==s_index_u){
                            weight2=(u_plane[l+1]-fminf(coord_u1,coord_u2))/fabsf(coord_u1-coord_u2);
                        }else if(l==e_index_u){
                            weight2=(fmaxf(coord_u1,coord_u2)-u_plane[l])/fabsf(coord_u1-coord_u2);
                        }else{
                            weight2=fabsf(du)/fabsf(coord_u1-coord_u2);
                        }
                        atomicAdd(&Dest[tid],Src[angle_ind*nu*nv+k*nu+l]*weight1*weight2*InterpWeight);
                    }                    
                }
            }
        }
    }
}

__global__ void flat_distance_backproj_arb(float* Dest, float* Src,float* x_plane,float* y_plane,float* z_plane,float* u_plane,float* v_plane,float* params,float SourceX, float SourceY, float SourceZ, float DetectorX,float DetectorY,float DetectorZ, float angle1, float angle2, int angle_ind){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int tid=y*gridDim.x*blockDim.x+x;
    float dx=params[0],dy=params[1],dz=params[2];
    int nx=(int)params[3],ny=(int)params[4],nz=(int)params[5],nu=(int)params[6],nv=(int)params[7];
    float du=params[8],dv=params[9];
    float R=params[18];
    float yc;
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
        x1=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y1=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z1=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x2=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y2=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z2=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x3=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y3=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z3=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x4=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y4=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z4=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x5=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y5=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z5=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x6=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y6=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z6=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x7=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y7=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z7=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x8=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y8=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z8=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);

        u_slope1=(SourceX-x1)/(SourceY-y1);
        u_slope2=(SourceX-x2)/(SourceY-y2);
        u_slope3=(SourceX-x3)/(SourceY-y3);
        u_slope4=(SourceX-x4)/(SourceY-y4);
        u_slope5=(SourceX-x5)/(SourceY-y5);
        u_slope6=(SourceX-x6)/(SourceY-y6);
        u_slope7=(SourceX-x7)/(SourceY-y7);
        u_slope8=(SourceX-x8)/(SourceY-y8);
        slope_min=fminf(u_slope1,fminf(u_slope2,fminf(u_slope3,fminf(u_slope4,fminf(u_slope5,fminf(u_slope6,fminf(u_slope7,u_slope8)))))));
        slope_max=fmaxf(u_slope1,fmaxf(u_slope2,fmaxf(u_slope3,fmaxf(u_slope4,fmaxf(u_slope5,fmaxf(u_slope6,fmaxf(u_slope7,u_slope8)))))));
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
        slope_min=fminf(v_slope1,fminf(v_slope2,fminf(v_slope3,fminf(v_slope4,fminf(v_slope5,fminf(v_slope6,fminf(v_slope7,v_slope8)))))));
        slope_max=fmaxf(v_slope1,fmaxf(v_slope2,fmaxf(v_slope3,fmaxf(v_slope4,fmaxf(v_slope5,fmaxf(v_slope6,fmaxf(v_slope7,v_slope8)))))));
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
                    weight1=1.0f;
                }else if(k==s_index_v){
                    weight1=(fmaxf(coord_v1,coord_v2)-v_plane[k+1])/fabsf(coord_v1-coord_v2);
                }else if(k==e_index_v){
                    weight1=(v_plane[k]-fminf(coord_v1,coord_v2))/fabsf(coord_v1-coord_v2);
                }else{
                    weight1=fabsf(dv)/fabsf(coord_v1-coord_v2);
                }
                for(l=s_index_u;l<=e_index_u;l++){
                    if(l>=0 && l<=nu-1){
                        if(s_index_u==e_index_u){
                            weight2=1.0f;
                        }else if(l==s_index_u){
                            weight2=(u_plane[l+1]-fminf(coord_u1,coord_u2))/fabsf(coord_u1-coord_u2);
                        }else if(l==e_index_u){
                            weight2=(fmaxf(coord_u1,coord_u2)-u_plane[l])/fabsf(coord_u1-coord_u2);
                        }else{
                            weight2=fabsf(du)/fabsf(coord_u1-coord_u2);
                        }
                        atomicAdd(&Dest[tid],Src[angle_ind*nu*nv+k*nu+l]*weight1*weight2*InterpWeight);
                    }                    
                }
            }
        }
    }
}

__global__ void curved_distance_backproj_arb(float* Dest, float* Src,float* x_plane,float* y_plane,float* z_plane,float* u_plane,float* v_plane,float* params,float SourceX, float SourceY, float SourceZ, float DetectorX,float DetectorY,float DetectorZ, float angle1, float angle2, int angle_ind){
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int tid=y*gridDim.x*blockDim.x+x;
    float dx=params[0],dy=params[1],dz=params[2];
    int nx=(int)params[3],ny=(int)params[4],nz=(int)params[5],nu=(int)params[6],nv=(int)params[7];
    float du=params[8],dv=params[9];
    float R=params[18];
    float yc;
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
        x1=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y1=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z1=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x2=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y2=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z2=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x3=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y3=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z3=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x4=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        y4=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        z4=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]-dz/2.0,angle1,angle2);
        
        x5=fx(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y5=fy(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z5=fz(x_plane[ix]+dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x6=fx(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y6=fy(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z6=fz(x_plane[ix]-dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x7=fx(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y7=fy(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z7=fz(x_plane[ix]+dx/2.0,y_plane[iy]-dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        
        x8=fx(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        y8=fy(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);
        z8=fz(x_plane[ix]-dx/2.0,y_plane[iy]+dy/2.0,z_plane[iz]+dz/2.0,angle1,angle2);

        u_slope1=(SourceX-x1)/(SourceY-y1);
        u_slope2=(SourceX-x2)/(SourceY-y2);
        u_slope3=(SourceX-x3)/(SourceY-y3);
        u_slope4=(SourceX-x4)/(SourceY-y4);
        u_slope5=(SourceX-x5)/(SourceY-y5);
        u_slope6=(SourceX-x6)/(SourceY-y6);
        u_slope7=(SourceX-x7)/(SourceY-y7);
        u_slope8=(SourceX-x8)/(SourceY-y8);
        slope_min=fminf(u_slope1,fminf(u_slope2,fminf(u_slope3,fminf(u_slope4,fminf(u_slope5,fminf(u_slope6,fminf(u_slope7,u_slope8)))))));
        slope_max=fmaxf(u_slope1,fmaxf(u_slope2,fmaxf(u_slope3,fmaxf(u_slope4,fmaxf(u_slope5,fmaxf(u_slope6,fmaxf(u_slope7,u_slope8)))))));
        coord_u1=-atan(slope_min);
        coord_u2=-atan(slope_max);
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
        slope_min=fminf(v_slope1,fminf(v_slope2,fminf(v_slope3,fminf(v_slope4,fminf(v_slope5,fminf(v_slope6,fminf(v_slope7,v_slope8)))))));
        slope_max=fmaxf(v_slope1,fmaxf(v_slope2,fmaxf(v_slope3,fmaxf(v_slope4,fmaxf(v_slope5,fmaxf(v_slope6,fmaxf(v_slope7,v_slope8)))))));
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
                    weight1=1.0f;
                }else if(k==s_index_v){
                    weight1=(fmaxf(coord_v1,coord_v2)-v_plane[k+1])/fabsf(coord_v1-coord_v2);
                }else if(k==e_index_v){
                    weight1=(v_plane[k]-fminf(coord_v1,coord_v2))/fabsf(coord_v1-coord_v2);
                }else{
                    weight1=fabsf(dv)/fabsf(coord_v1-coord_v2);
                }
                for(l=s_index_u;l<=e_index_u;l++){
                    if(l>=0 && l<=nu-1){
                        if(s_index_u==e_index_u){
                            weight2=1.0f;
                        }else if(l==s_index_u){
                            weight2=(u_plane[l+1]-fminf(coord_u1,coord_u2))/fabsf(coord_u1-coord_u2);
                        }else if(l==e_index_u){
                            weight2=(fmaxf(coord_u1,coord_u2)-u_plane[l])/fabsf(coord_u1-coord_u2);
                        }else{
                            weight2=fabsf(du)/fabsf(coord_u1-coord_u2);
                        }
                        atomicAdd(&Dest[tid],Src[angle_ind*nu*nv+k*nu+l]*weight1*weight2*InterpWeight);
                    }                    
                }
            }
        }
    }
}
    
    """)
    return mod
