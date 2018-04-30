from Reconstruction import Reconstruction
import numpy as np
import glob, sys, os,copy
import logging
import matplotlib.pyplot as plt

''' global variables'''
pi = np.pi
sqrt=np.sqrt
eps=1e-7
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

''' gradient of total variation'''
def tv_grad2(image):
    det=0.006
    w1=image[1:-1,1:-1,1:-1]-image[0:-2,1:-1,1:-1]
    w2=image[1:-1,1:-1,1:-1]-image[1:-1,0:-2,1:-1]
    w3=image[1:-1,1:-1,1:-1]-image[1:-1,1:-1,0:-2]
    e1=np.exp(-1*(w1/det)**2)
    e2=np.exp(-1*(w2/det)**2)
    e3=np.exp(-1*(w3/det)**2)
    t1=(e1*w1+e2*w2+e3*w3)/(sqrt(eps+e1*w1**2*+e2*w2**2+e3*w3**2))
    w4=image[2:,1:-1,1:-1]-image[1:-1,1:-1,1:-1]
    w5=image[2:,1:-1,1:-1]-image[2:,0:-2,1:-1]
    w6=image[2:,1:-1,1:-1]-image[2:,1:-1,0:-2]
    e4=np.exp(-1*(w4/det)**2)
    e5=np.exp(-1*(w5/det)**2)
    e6=np.exp(-1*(w6/det)**2)
    t2=(-e4*w4)/(sqrt(eps+e4*w4**2+e5*w5**2+e6*w6**2))
    w7=image[1:-1,2:,1:-1]-image[0:-2,2:,1:-1]
    w8=image[1:-1,2:,1:-1]-image[1:-1,1:-1,1:-1]
    w9=image[1:-1,2:,1:-1]-image[1:-1,2:,0:-2]
    e7=np.exp(-1*(w7/det)**2)
    e8=np.exp(-1*(w8/det)**2)
    e9=np.exp(-1*(w9/det)**2)
    t3=(-e8*w8)/(sqrt(eps+e7*w7**2+e8*w8**2+e9*w9**2))
    w10=image[1:-1,1:-1,2:]-image[0:-2,1:-1,2:]
    w11=image[1:-1,1:-1,2:]-image[1:-1,0:-2,2:]
    w12=image[1:-1,1:-1,2:]-image[1:-1,1:-1,1:-1]
    e10=np.exp(-1*(w10/det)**2)
    e11=np.exp(-1*(w11/det)**2)
    e12=np.exp(-1*(w12/det)**2)
    t4=(-e12*w12)/(sqrt(eps+e10*w10**2+e11*w11**2+e12*w12**2))
    dtv=t1+t2+t3+t4
    tv_norm=sqrt(np.sum(dtv**2))
    #if(np.abs(tv_norm)<eps): tv_norm=1.0
    dtv/=tv_norm
    dtv=np.lib.pad(dtv,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
    return dtv
def tv_grad(image):
    w1=image[1:-1,1:-1,1:-1]-image[0:-2,1:-1,1:-1]
    w2=image[1:-1,1:-1,1:-1]-image[1:-1,0:-2,1:-1]
    w3=image[1:-1,1:-1,1:-1]-image[1:-1,1:-1,0:-2]
    t1=(w1+w2+w3)/(sqrt(eps+w1**2+w2**2+w3**2))
    w4=image[2:,1:-1,1:-1]-image[1:-1,1:-1,1:-1]
    w5=image[2:,1:-1,1:-1]-image[2:,0:-2,1:-1]
    w6=image[2:,1:-1,1:-1]-image[2:,1:-1,0:-2]
    t2=(-w4)/(sqrt(eps+w4**2+w5**2+w6**2))
    w7=image[1:-1,2:,1:-1]-image[0:-2,2:,1:-1]
    w8=image[1:-1,2:,1:-1]-image[1:-1,1:-1,1:-1]
    w9=image[1:-1,2:,1:-1]-image[1:-1,2:,0:-2]
    t3=(-w8)/(sqrt(eps+w7**2+w8**2+w9**2))
    w10=image[1:-1,1:-1,2:]-image[0:-2,1:-1,2:]
    w11=image[1:-1,1:-1,2:]-image[1:-1,0:-2,2:]
    w12=image[1:-1,1:-1,2:]-image[1:-1,1:-1,1:-1]
    t4=(-w12)/(sqrt(eps+w10**2+w11**2+w12**2))
    dtv=t1+t2+t3+t4
    tv_norm=sqrt(np.sum(dtv**2))
    if(np.abs(tv_norm)<eps): tv_norm=1.0
    dtv/=tv_norm
    dtv=np.lib.pad(dtv,((1,1),(1,1),(1,1)),'constant',constant_values=((0,0),(0,0),(0,0)))
    return dtv
''' POCS step'''
def POCS(R,proj0,norm1,norm2,beta,na):
    #f0=np.copy(R.image)
    for i in range(na):
        recon_tmp=np.copy(R.image)
        R.forward()
        #g0=np.copy(R.proj)
        R.proj=(proj0-R.proj)/(norm2.proj+eps)
        R.proj[np.where(norm2.proj==0)]=0
        R.backward()
        R.image[np.where(norm1.image==0)]=0
        R.image=recon_tmp+beta*(R.image/(norm1.image+eps))
        R.image[np.where(R.image<0)]=0
        #R.SaveRecon('POCS_update_%04d.dat'%i)
    return R
    #R.forward()
    #g_new=np.copy(R.proj)
    #f_new=np.copy(R.image)
    #dd=sqrt(np.sum((g0-g_new)**2))
    #da=sqrt(np.sum((f0-f_new)**2))
    #return R
''' TV step'''
def TV(image,dtvg,ng):
    for i in range(ng):
        log.debug('%d-th tv minimization'%i)
        dtv=tv_grad2(image)
        #dtv.tofile('TV_grad_%04d.dat'%i,sep='',format='')
        image=image-dtvg*dtv
    return image
    #return R
# data = np.fromfile('Shepp_Logal_3d_256.dat', dtype=np.float32).reshape([256, 256, 256])

params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
          'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.78125, 0.78125],
          'NumberOfViews': 180, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [512, 512, 512],
          'PhantomCenter': [0, 0, 0], 'Origin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'ram-lak', 
          'cutoff': 0.3, 'GPU': 1}
#params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0,
#          'EndAngle': 2 * pi, 'NumberOfDetectorPixels': [512, 384], 'DetectorPixelSize': [0.5, 0.5],
#          'NumberOfViews': 45, 'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [256, 256, 256],
#          'PhantomCenter': [0, 0, 0], 'Origin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'ram-lak', 
#          'cutoff': 0.3, 'GPU': 1}
R = Reconstruction(params)
#filename = 'Shepp_Logan_3d_256.dat'
filename='../Projection/R_004/full_recon.dat'
#filename='/home/leehoy/CTReconstruction/python/Shepp_Logan_3d_256.dat'

R.LoadRecon(filename, params['NumberOfImage'])
ph = np.copy(R.image)
R.forward()
log.debug(R.proj.shape)
#log.debug(R.proj.dtype)
proj0 = np.copy(R.proj)

R.Filtering()
R.backward()

norm1 = Reconstruction(params)
norm1.proj = np.ones(
    [params['NumberOfViews'], params['NumberOfDetectorPixels'][1], params['NumberOfDetectorPixels'][0]],
    dtype=np.float32)
norm1.backward()
#norm1.SaveRecon('Norm1.dat')
norm2 = Reconstruction(params)
norm2.image = np.ones(params['NumberOfImage'],dtype=np.float32)
norm2.forward()
#norm2.SaveProj('nomr2.dat')
assert( (norm2.proj>=0).all())
assert( (norm1.image>=0).all())
Niter = 20
rmse = np.zeros(Niter, dtype=np.float32)
beta=1.0
beta_red=0.999
na=5
ng=20
alpha=0.007
r_max=0.99
alpha_red=0.95

for i in range(Niter):
    log.info('%d-th iteration'%i)
    f0=np.copy(R.image)
    #R=POCS(R,norm1,norm2.beta,na)
    for a in range(na):
        recon_tmp=np.copy(R.image)
        R.forward()
        R.proj=(proj0-R.proj)/(norm2.proj+eps)
        R.proj[np.where(norm2.proj==0)]=0
        R.backward()
        R.image[np.where(norm1.image==0)]=0
        R.image=recon_tmp+beta*R.image/(norm1.image+eps)
        R.image[np.where(R.image<0)]=0
    R.SaveRecon('R_004_ART_%04d.dat'%i)
    #f_res=np.copy(R.image)
    #dd=np.sqrt(np.sum((R.proj-proj0)**2))
    dp=np.sqrt(np.sum((R.image-f0)**2)) # adist
    dtvg=alpha*dp
    #if(i==0):
    #    dtvg=alpha*dp
    f0=np.copy(R.image)
    for g in range(ng):
       dtv=tv_grad(R.image)
       R.image=R.image-dtvg*dtv
    #R.image=np.copy(TV(f0,dtvg,ng))
    dg=np.sqrt(np.sum((R.image-f0)**2)) # gdist
    #R.forward()
    if(dg>r_max*dp):
        alpha*=alpha_red
    beta*=beta_red
    log.debug('dtvg: %f beta: %f dp: %f dg: %f'%(dtvg,beta,dp,dg))
    #log.debug(sqrt(np.sum((R.proj-proj0)**2)))
    rmse[i]=sqrt(np.mean((R.image-ph)**2))
    log.debug(rmse[i])
    R.SaveRecon('R_004_TV2_POCS_%04d.dat'%i)
R.SaveRecon('R_004_TV2_POCS.dat')
