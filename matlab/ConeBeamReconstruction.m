%% Reconstruction starts
% To Do: Half fan reconstruction
%        Acceleration
% Done:  Filter implementation using sinc function
%        Gpu implementation using gpuArray - feels like very slow
%        Compared with FDK code result
%        Filter selection
% ConeBeamLoad;
tic;
close all;
R=SourceToAxis; % Dso
D=SourceToDetector-SourceToAxis;
nx=DetectorWidth;
ny=DetectorHeight;
ki=(1:nx)-(nx-1)/2;
p=(1:ny)-(ny-1)/2;
ki=ki*DetectorPixelWidth;
p=p*DetectorPixelHeight;
ZeroPaddedLength=2^nextpow2(2*(nx-1));
deltaS=DetectorPixelWidth*R/(D+R);
gpuFlag=0;
cutoff=0.3;
filter=FilterLine(ZeroPaddedLength+1,deltaS,'hann',cutoff);
Recon=zeros(ReconX,ReconY,ReconZ);
DetectorSize=[nx*DetectorPixelWidth,ny*DetectorPixelHeight];
fov=2*R*sin(atan(DetectorSize(1)/2/(D+R)));
fovz=2*R*sin(atan(DetectorSize(2)/2/(D+R)));
x=linspace(-fov/2,fov/2,ReconX);
y=linspace(-fov/2,fov/2,ReconY);
z=linspace(-fovz/2,fovz/2,ReconZ);
[xx,yy]=meshgrid(x,y);
ProjectionAngle=linspace(0,AngleCoverage,NumberOfViews+1);
ProjectionAngle=ProjectionAngle(1:end-1);
ki=(ki*R)./(R+D);
p=(p*R)./(R+D);
[pp,kk]=meshgrid(p,ki);
weight=R./(sqrt(R^2+kk.^2+pp.^2));
dtheta=ProjectionAngle(2)-ProjectionAngle(1);

for i=1:NumberOfViews
    fprintf('%d\n',i);
    WeightedProjection=weight.*Projection(:,:,i);
    Q=zeros(size(Projection(:,:,i)));
    angle=ProjectionAngle(i);
    for k=1:DetectorHeight
        tmp=real(ifft(ifftshift(fftshift(fft(WeightedProjection(:,k),ZeroPaddedLength)).*filter)));
        Q(:,k)=tmp(1:nx);%*deltaS;
    end
    t=xx.*cos(angle)+yy.*sin(angle);
    s=-xx.*sin(angle)+yy.*cos(angle);
    if gpuFlag==1
        Qgpu=gpuArray(Q);
        for l=1:ReconZ
            InterpX=(R.*t)./(R-s);
            InterpY=(R.*z(l))./(R-s);
            InterpW=(R^2)./(R-s).^2;
            InterpXGpu=gpuArray(InterpX);
            InterpYGpu=gpuArray(InterpY);
            vq=interp2(p,ki,Qgpu,InterpYGpu,InterpXGpu,'linear',0); % interp2 using gpu
            tmp=reshape(gather(vq),[ReconX ReconY]);
            Recon(:,:,l)=Recon(:,:,l)+InterpW.*tmp*dtheta;
        end
    else
        for l=256:256
%         for l=1:ReconZ
            InterpX=(R.*t)./(R-s);
            InterpY=(R.*z(l))./(R-s);
            InterpW=(R^2)./(R-s).^2;
            vq=interp2(p,ki,Q,InterpY,InterpX,'spline',0);
            Recon(:,:,l)=Recon(:,:,l)+InterpW.*vq*dtheta;
%             imshow(Recon(:,:,l),[]);
        end
    end
    
end
toc
% f=fopen('ConeBeamReconstructed.dat','w');
% fwrite(f,Recon,precision);
% fclose(f);
% CompareCBCTRecon; 