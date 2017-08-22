tic;
nx=256;
ny=nx;
nz=nx;
ph=phantom3d(nx);
Source_init=[0,1000,0]; % Initial source position
Detector_init=[0,-500,0]; % Initial detector position
Origin=[0,0,0]; % Rotating center
SAD=sqrt(sum((Source_init-Origin).^2));
SDD=sqrt(sum((Source_init-Detector_init).^2));
DetectorPixelSizeH=0.5; % Detector pixel spacing
DetectorPixelSizeV=0.5; % Detector pixel spacing
NumberOfDetectorPixels=[512 ,384]; % Number of detector rows and chnnels
PhantomCenter=[0,0,0]; % Center of phantom
PhantomPixelSpacingX=0.5;
PhantomPixelSpacingY=0.5;
PhantomPixelSpacingZ=0.5;
dx=PhantomPixelSpacingX; %phantom pixel spacing
dy=-PhantomPixelSpacingY;
dz=-PhantomPixelSpacingZ;
nTheta=90;
StartAngle=0;
EndAngle=2*pi;

tol_max=1e6;
tol_min=1e-6;
Xplane=(PhantomCenter(1)-size(ph,1)/2+(0:nx))*dx;
Yplane=(PhantomCenter(2)-size(ph,2)/2+(0:ny))*dy;
Zplane=(PhantomCenter(3)-size(ph,3)/2+(0:nz))*dz;
Xplane=Xplane-dx/2;
Yplane=Yplane-dy/2;
Zplane=Zplane-dz/2;
theta=linspace(StartAngle,EndAngle,nTheta+1);
theta=theta(1:end-1);
proj=zeros(NumberOfDetectorPixels(1),NumberOfDetectorPixels(2),nTheta);

% Rotating CCW direction starting from x-axis
% TO Dos:
%   Add direction configurations
%   Expand to cone-beam projection -> Solved
%   Reduce discontinuity between angles - this is cause by indexing.
%       From zero to 90 degrees and 270 to 360 degrees, detector has larger number of pixels in
%       bottom/right position from the center and it changes to up/left
%       between 90 to 270 degrees -> Solved

for angle_index=1:nTheta
    SourceX=-SAD*sin(theta(angle_index)); % source coordinate
    SourceY=SAD*cos(theta(angle_index));
    SourceZ=0; % source position is not flutuating in currently
    DetectorX=(SDD-SAD)*sin(theta(angle_index));  % center of detector coordinate
    DetectorY=-(SDD-SAD)*cos(theta(angle_index));
    DetectorZ=0;
    DetectorLengthH=(floor(-NumberOfDetectorPixels(1)/2):floor(NumberOfDetectorPixels(1)/2))*DetectorPixelSizeH; %horizontal detector length
    DetectorLengthV=(floor(-NumberOfDetectorPixels(2)/2):floor(NumberOfDetectorPixels(2)/2))*DetectorPixelSizeV; %horizontal detector length
    if(abs(tan(theta(angle_index)))<=tol_min)
        DetectorIndex=[DetectorX+DetectorLengthH; repmat(DetectorY,1,size(DetectorLengthH,2))];
        DetectorIndexZ=DetectorZ-DetectorLengthV;
    elseif(tan(theta(angle_index))>=tol_max)
        DetectorIndex=[repmat(DetectorX,1,size(DetectorLengthH,2)); DetectorY+DetectorLengthH];
        DetectorIndexZ=DetectorZ-DetectorLengthV;
    else
        xx=sqrt(DetectorLengthH.^2./(1+tan(theta(angle_index))^2));
        yy=tan(theta(angle_index))*sqrt(DetectorLengthH.^2./(1+tan(theta(angle_index))^2));
        DetectorIndex=[DetectorX+sign(DetectorLengthH).*xx;...
            DetectorY+sign(DetectorLengthH).*yy];
        DetectorIndexZ=DetectorZ-DetectorLengthV; % to make upper pixels come first
    end
    if(DetectorY>0)
        DetectorIndex=DetectorIndex(:,end:-1:1);
    end
    DetectorIndex=DetectorIndex(:,1:end-1); % The index pointing center of detector pixels
    DetectorIndexZ=DetectorIndexZ(1:end-1);
    xy=repmat(DetectorIndex,1,1,NumberOfDetectorPixels(2));
    z=reshape(repmat(DetectorIndexZ,NumberOfDetectorPixels(1),1),[1,NumberOfDetectorPixels(1),...
        NumberOfDetectorPixels(2)]);
    DetectorIndex=[xy;z];
    for detector_index_v=1:size(DetectorIndex,3)
        for detector_index_h=1:size(DetectorIndex,2)
            alpha_x=(Xplane-SourceX)/(DetectorIndex(1,detector_index_h,detector_index_v)...
                -SourceX);
            alpha_y=(Yplane-SourceY)/(DetectorIndex(2,detector_index_h,detector_index_v)...
                -SourceY);
            alpha_z=(Zplane-SourceZ)/(DetectorIndex(3,detector_index_h,detector_index_v)...
                -SourceZ);
            alpha_min=max([0,min(alpha_x(1),alpha_x(end)),min(alpha_y(1),alpha_y(end)),...
                min(alpha_z(1),alpha_z(end))]);
            alpha_max=min([1,max(alpha_x(1),alpha_x(end)),max(alpha_y(1),alpha_y(end)),...
                max(alpha_z(1),alpha_z(end))]);
            if(alpha_min>alpha_max || abs(alpha_min-alpha_max)<tol_min)
                %put zeros if the ray doesn't intersect with phantom
                proj(detector_index_h,detector_index_v,angle_index)=0;
            else
                if(abs(SourceX-DetectorIndex(1,detector_index_h,detector_index_v))<tol_min)
                    alpha_x=[];
                elseif(SourceX<DetectorIndex(1,detector_index_h,detector_index_v))
                    i_min=ceil((nx+1)-(Xplane(end)-alpha_min*(DetectorIndex(1,detector_index_h,detector_index_v)...
                        -SourceX)-SourceX)/dx);
                    i_max=floor(1+(SourceX+alpha_max*(DetectorIndex(1,detector_index_h,detector_index_v)...
                        -SourceX)-Xplane(1))/dx);
                    alpha_x=alpha_x(i_min:i_max);
                else
                    i_min=ceil((nx+1)-(Xplane(end)-alpha_max*(DetectorIndex(1,detector_index_h,detector_index_v)...
                        -SourceX)-SourceX)/dx);
                    i_max=floor(1+(SourceX+alpha_min*(DetectorIndex(1,detector_index_h,detector_index_v)...
                        -SourceX)-Xplane(1))/dx);
                    alpha_x=alpha_x(i_max:-1:i_min);
                end
                if(abs(SourceY-DetectorIndex(2,detector_index_h,detector_index_v))<tol_min)
                    alpha_y=[];
                elseif(SourceY>DetectorIndex(2,detector_index_h,detector_index_v))
                    j_min=ceil((ny+1)-(Yplane(end)-alpha_min*(DetectorIndex(2,detector_index_h,detector_index_v)...
                        -SourceY)-SourceY)/dy);
                    j_max=floor(1+(SourceY+alpha_max*(DetectorIndex(2,detector_index_h,detector_index_v)...
                        -SourceY)-Yplane(1))/dy);
                    alpha_y=alpha_y(j_min:j_max);
                else
                    j_min=ceil((ny+1)-(Yplane(end)-alpha_max*(DetectorIndex(2,detector_index_h,detector_index_v)...
                        -SourceY)-SourceY)/dy);
                    j_max=floor(1+(SourceY+alpha_min*(DetectorIndex(2,detector_index_h,detector_index_v)...
                        -SourceY)-Yplane(1))/dy);
                    alpha_y=alpha_y(j_max:-1:j_min);
                end
                if(abs(SourceZ-DetectorIndex(3,detector_index_h,detector_index_v))<tol_min)
                    alpha_z=[];
                elseif(SourceZ>DetectorIndex(3,detector_index_h,detector_index_v))
                    k_min=ceil((nz+1)-(Zplane(end)-alpha_min*(DetectorIndex(3,detector_index_h,detector_index_v)...
                        -SourceZ)-SourceZ)/dz);
                    k_max=floor(1+(SourceZ+alpha_max*(DetectorIndex(3,detector_index_h,detector_index_v)...
                        -SourceZ)-Zplane(1))/dz);
                    alpha_z=alpha_z(k_min:k_max);
                else
                    k_min=ceil((nz+1)-(Zplane(end)-alpha_max*(DetectorIndex(3,detector_index_h,detector_index_v)...
                        -SourceZ)-SourceZ)/dz);
                    k_max=floor(1+(SourceZ+alpha_min*(DetectorIndex(3,detector_index_h,detector_index_v)...
                        -SourceZ)-Zplane(1))/dz);
                    alpha_z=alpha_z(k_max:-1:k_min);
                end
                alpha=uniquetol(sort([alpha_min,alpha_x,alpha_y,alpha_z,alpha_max]),tol_min/alpha_max);
                l=zeros(length(alpha)-1,1);
                d12=sqrt((SourceX-DetectorIndex(1,detector_index_h,detector_index_v))^2+...
                (SourceY-DetectorIndex(2,detector_index_h,detector_index_v))^2+...
                    (SourceZ-DetectorIndex(3,detector_index_h,detector_index_v))^2);
                for i=1:length(l)
                    l(i)=d12*(alpha(i+1)-alpha(i));
                end
                index=zeros(length(l),2);
                for i=1:size(index,1)
%                     l=d12*(alpha(i+1)-alpha(i));
                    alpha_mid=(alpha(i+1)+alpha(i))/2;
                    xx=(SourceX+alpha_mid*(DetectorIndex(1,detector_index_h,detector_index_v)...
                        -SourceX)-Xplane(1))/dx;
                    yy=(SourceY+alpha_mid*(DetectorIndex(2,detector_index_h,detector_index_v)...
                        -SourceY)-Yplane(1))/dy;
                    zz=(SourceZ+alpha_mid*(DetectorIndex(3,detector_index_h,detector_index_v)...
                        -SourceZ)-Zplane(1))/dz;
                    if(abs(xx)<=tol_min)
                        xx=0;
                    end
                    if(abs(yy)<=tol_min)
                        yy=0;
                    end
                    if(abs(zz)<=tol_min)
                        zz=0;
                    end
                    index(i,1)=floor(1+xx);
                    index(i,2)=floor(1+yy);
                    index(i,3)=floor(1+zz);
%                     proj(detector_index_h,detector_index_v,angle_index)=...
%                         proj(detector_index_h,detector_index_v,angle_index)...
%                         +l(i)*ph(floor(1+xx),floor(1+yy),floor(1+zz));
                end
                for i=1:length(l)
                    proj(detector_index_h,detector_index_v,angle_index)=...
                        proj(detector_index_h,detector_index_v,angle_index)...
                        +l(i)*ph(index(i,1),index(i,2),index(i,3));
                end
            end
        end
    end
end
% imagesc(proj);
% colormap gray;
f=fopen('Proj_siddons3D.dat','w');
fwrite(f,proj,'float32');
fclose(f);
toc