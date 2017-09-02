function [ recon ] = DistanceDrivenBackprojection2D( proj )
%UNTITLED6 이 함수의 요약 설명 위치
%   자세한 설명 위치
    N=size(proj,1);
    M=size(proj,2);
%     DetectorWidth=N;
    R=1000;
    D=500;
    r_spacing=0.5;
    deltaS=r_spacing*R/(R+D);
    SourceCenter=[-1000, 0];
    DetectorCenter=[500,0];
    nx=256;
    ny=nx;
%     DetectorSize=r_spacing*DetectorWidth;
    gamma=((0:N-1)-(N-1)/2)*deltaS;
    ZeroPaddedLength=2^nextpow2(2*(N-1));
    cutoff=0.3;
    FilterType='hann';
    filter=FilterLine(ZeroPaddedLength+1,deltaS,FilterType,cutoff)*0.5;
%     fov=2*R*sin(atan((DetectorSize/2)/(R+D)));
    ReconSpacingX=0.5; % fov/nx;
    ReconSpacingY=-0.5;
    recon_planeX=(-nx/2+(0:nx))*ReconSpacingX; % pixel boundaries of image
    recon_planeY=(-ny/2+(0:ny))*ReconSpacingY;
    recon_planeX=recon_planeX-ReconSpacingX/2;
    recon_planeY=recon_planeY-ReconSpacingY/2;
%     x=(-(nx-1)/2:(nx-1)/2)*ReconSpacingX;
%     y=(-(ny-1)/2:(ny-1)/2)*ReconSpacingY;
%     [X,Y]=meshgrid(x,y);
%     xpr=X;
%     ypr=Y;
    recon=zeros(nx,ny);
%     [phi,r]=cart2pol(xpr,ypr);
    theta=linspace(0,360,M+1);
    theta=theta*(pi/180);
    dtheta=(pi*2)/M;
    for i=1:M
        R1=proj(:,i);
        w=((R)./sqrt((R)^2+gamma'.^2));
        angle=theta(i);
        R2=w.*R1;
        Q=real(ifft(ifftshift(fftshift(fft(R2,ZeroPaddedLength)).*filter)));
        Q=Q(1:length(R2));
%         fprintf('%d\n',i);
%         [DetectorBoundary1,DetectorBoundary2]=DetectorBoundaryFinder(N,r_spacing,angle,D);
        recon=recon+backproj(Q,recon_planeX,recon_planeY,angle,R,R+D,N,r_spacing,SourceCenter,DetectorCenter)*dtheta;
    end
    close all;
    imshow(recon,[]);
end
function bp=backproj(proj,recon_planeX,recon_planeY,angle,SAD,SDD,nd,DetectorPixelSpacing,Source,Detector)
    nx=length(recon_planeX)-1;
    ny=length(recon_planeY)-1;
    SourceX=Source(1);
    SourceY=Source(2);
    DetectorX=Detector(1);
    DetectorY=Detector(2);
    bp=zeros(nx,ny);
    dx=recon_planeX(2)-recon_planeX(1);
    dy=recon_planeY(2)-recon_planeY(1);
    recon_PixelsX=recon_planeX(1:end-1)+dx/2; %x center of the pixels
    recon_PixelsY=recon_planeY(1:end-1)+dy/2; %y center of the pixels
    for i=1:nx
        for j=1:ny
            xr=recon_PixelsX(i)*cos(angle)+recon_PixelsY(j)*sin(angle);
            yr=-recon_PixelsX(i)*sin(angle)+recon_PixelsY(j)*cos(angle);
%             not exact detector contribution making noisy signal?
            n_min=((yr-0.25)/(xr+SAD))*SDD/DetectorPixelSpacing+nd/2;
            n_max=((yr+0.25)/(xr+SAD))*SDD/DetectorPixelSpacing+nd/2;
            
            for k=floor(n_min):floor(n_max)
                if(k<1 || k>nx)
                    continue;
                end
                y1=DetectorY+(k-nd/2)*DetectorPixelSpacing;
                slope=(y1-SourceY)/(DetectorX-SourceX);
                intercept=slope*xr-yr;
                a=sqrt(slope^2/(1+slope^2));
                b=-1*sign(slope)*sqrt(1-a^2);
                c=b*intercept;
                a2=abs(a);
                b2=abs(b);
                d=abs(a*xr+b*yr+c)/sqrt(a^2+b^2);
                d1=abs(a2-b2)/2;
                d2=(a2+b2)/2;
                if(d<d1 && a2<b2)
                    l=1/b2;
                elseif(d<d1&& a2>=b2)
                    l=1/a2;
                elseif(d>=d1&& d<d2)
                    l=(d2-d)/(a2*b2);
                elseif(d>=d2)
                    l=0;
                end
                l=1;
                if(k==floor(n_min))
                    bp(i,j)=proj(k)*1*l*(ceil(n_min)-n_min)/(n_max-n_min);
                elseif(k>floor(n_min) && k<floor(n_max))
                    bp(i,j)=proj(k)*1*l/(n_max-n_min);
                elseif(k==floor(n_max))
                    bp(i,j)=proj(k)*1*l*(n_max-floor(n_max))/(n_max-n_min);
                else
                    fprintf('????\n');
                end
            end
        end
    end
end
function [DetectorBoundary1,DetectorBoundary2]=DetectorBoundaryFinder(NumberOfDetectorPixels,...
    DetectorPixelSize,angle,Distance)
    DetectorX=Distance*sin(angle);  % center of detector coordinate
    DetectorY=-Distance*cos(angle);
    DetectorLength=(floor(-NumberOfDetectorPixels(1)/2):floor(NumberOfDetectorPixels(1)/2))*DetectorPixelSize;
    if(abs(tan(theta(angle_index)))<=1e-6) % detector is parallel to x-axis
        DetectorIndex=[DetectorX+DetectorLength; repmat(DetectorY,1,size(DetectorLength,2))];
    elseif(tan(theta(angle_index))>=1e6) % detector is parallel to y-axis
        DetectorIndex=[repmat(DetectorX,1,size(DetectorLength,2)); DetectorY+DetectorLength];
    else
        xx=sqrt(DetectorLength.^2./(1+tan(theta(angle_index))^2));
        yy=tan(theta(angle_index))*sqrt(DetectorLength.^2./(1+tan(theta(angle_index))^2));
        DetectorIndex=[DetectorX+sign(DetectorLength).*xx;...
            DetectorY+sign(DetectorLength).*yy];
    end
    if(DetectorY>0)
        DetectorIndex=DetectorIndex(:,end:-1:1);
    end
    DetectorIndex=DetectorIndex(:,1:end-1); % The index pointing center of detector pixels
    DetectorBoundary1=[DetectorIndex(1,detector_index)-cos(theta(angle_index))*...
                DetectorPixelSize/2,DetectorIndex(2,detector_index)-sin(theta(angle_index))*...
                DetectorPixelSize/2];
    DetectorBoundary2=[DetectorIndex(1,detector_index)+cos(theta(angle_index))*...
        DetectorPixelSize/2,DetectorIndex(2,detector_index)+sin(theta(angle_index))*...
        DetectorPixelSize/2];
end