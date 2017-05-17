tic;

nx=256;
ny=nx;
ph=phantom(nx);
Source_init=[0,1000]; % Initial source position
Detector_init=[0,-500]; % Initial detector position
Origin=[0,0]; % Rotating center
SAD=sqrt(sum((Source_init-Origin).^2));
SDD=sqrt(sum((Source_init-Detector_init).^2));
DetectorPixelSize=1; % Detector pixel spacing
NumberOfDetectorPixels=[1024 ,1]; % Number of detector rows and chnnels
PhantomCenter=[0,0]; % Center of phantom
dx=1; %phantom pixel spacing
dy=1;
nTheta=720;
StartAngle=0;
EndAngle=2*pi;


Xplane=PhantomCenter(1)-size(ph,1)/2+(0:nx-1)*dx;
Yplane=PhantomCenter(2)-size(ph,2)/2+(0:ny-1)*dy;
theta=linspace(StartAngle,EndAngle,nTheta+1);
theta=theta(1:end-1);
proj=zeros(NumberOfDetectorPixels(1),nTheta);

% Rotating CCW direction starting from x-axis
% TO Dos:
%   Add direction configurations
%   Expand to cone-beam projection
%   Reduce discontinuity between angles - this is cause by indexing.
%       From zero to 90 degrees and 270 to 360 degrees, detector has larger number of pixels in
%       bottom/right position from the center and it changes to up/left
%       between 90 to 270 degrees -> Solved

for angle_index=1:nTheta
    SourceX=-SAD*sin(theta(angle_index)); % source coordinate
    SourceY=SAD*cos(theta(angle_index));
    DetectorX=(SDD-SAD)*sin(theta(angle_index));  % center of detector coordinate
    DetectorY=-(SDD-SAD)*cos(theta(angle_index));
    DetectorLength=(floor(-NumberOfDetectorPixels(1)/2):floor(NumberOfDetectorPixels(1)/2))*DetectorPixelSize;
    if(abs(tan(theta(angle_index)))<=1e-6)
        DetectorIndex=[DetectorX+DetectorLength; repmat(DetectorY,1,size(DetectorLength,2))];
    elseif(tan(theta(angle_index))>=1e6)
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
    for detector_index=1:size(DetectorIndex,2)
        alpha_x=(Xplane-SourceX)/(DetectorIndex(1,detector_index)-SourceX);
        alpha_y=(Yplane-SourceY)/(DetectorIndex(2,detector_index)-SourceY);
        alpha_min=max([0,min(alpha_x(1),alpha_x(end)),min(alpha_y(1),alpha_y(end))]);
        alpha_max=min([1,max(alpha_x(1),alpha_x(end)),max(alpha_y(1),alpha_y(end))]);
        if(alpha_min>=alpha_max)
            continue;
        end
        if(SourceX==DetectorIndex(1,detector_index))
            alpha_x=[];
        elseif(SourceX<DetectorIndex(1,detector_index))
            i_min=ceil(nx-(Xplane(end)-alpha_min*(DetectorIndex(1,detector_index)-SourceX)-SourceX)/dx);
            i_max=floor(1+(SourceX+alpha_max*(DetectorIndex(1,detector_index)-SourceX)-Xplane(1))/dx);
            alpha_x=alpha_x(i_min:i_max);
        else
            i_min=ceil(nx-(Xplane(end)-alpha_max*(DetectorIndex(1,detector_index)-SourceX)-SourceX)/dx);
            i_max=floor(1+(SourceX+alpha_min*(DetectorIndex(1,detector_index)-SourceX)-Xplane(1))/dx);
            alpha_x=alpha_x(i_max:-1:i_min);
        end
        if(SourceY==DetectorIndex(2,detector_index))
            alpha_y=[];
        elseif(SourceY<DetectorIndex(2,detector_index))
            j_min=ceil(ny-(Yplane(end)-alpha_min*(DetectorIndex(2,detector_index)-SourceY)-SourceY)/dy);
            j_max=floor(1+(SourceY+alpha_max*(DetectorIndex(2,detector_index)-SourceY)-Yplane(1))/dy);
            alpha_y=alpha_y(j_min:j_max);
        else
            j_min=ceil(ny-(Yplane(end)-alpha_max*(DetectorIndex(2,detector_index)-SourceY)-SourceY)/dy);
            j_max=floor(1+(SourceY+alpha_min*(DetectorIndex(2,detector_index)-SourceY)-Yplane(1))/dy);
            alpha_y=alpha_y(j_max:-1:j_min);
        end
        alpha=unique(sort([alpha_min,alpha_x,alpha_y,alpha_max]));
        l=zeros(length(alpha)-1,1);
        d12=sqrt((SourceX-DetectorIndex(1,detector_index))^2+(SourceY-DetectorIndex(2,detector_index))^2);
        for i=1:length(l)
            l(i)=d12*(alpha(i+1)-alpha(i));
        end
        index=zeros(length(l),2);
        for i=1:size(index,1)
            alpha_mid=(alpha(i+1)+alpha(i))/2;
            xx=(SourceX+alpha_mid*(DetectorIndex(1,detector_index)-SourceX)-Xplane(1))/dx;
            yy=(SourceY+alpha_mid*(DetectorIndex(2,detector_index)-SourceY)-Yplane(1))/dy;
            if(abs(xx)<=1e-7)
                xx=0;
            end
            if(abs(yy)<=1e-7)
                yy=0;
            end
            index(i,1)=floor(1+xx);
            index(i,2)=floor(1+yy);
        end
        for i=1:length(l)
            proj(detector_index,angle_index)=proj(detector_index,angle_index)+l(i)*ph(index(i,1),index(i,2));
        end
    end
end
imagesc(proj);
colormap gray;
toc