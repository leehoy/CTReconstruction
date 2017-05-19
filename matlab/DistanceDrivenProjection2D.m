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

Xplane=PhantomCenter(1)-size(ph,1)/2+(0:nx)*dx; % pixel boundaries of image
Yplane=PhantomCenter(2)-size(ph,2)/2+(0:ny)*dy;
theta=linspace(StartAngle,EndAngle,nTheta+1);
theta=theta(1:end-1);
proj=zeros(NumberOfDetectorPixels(1),nTheta);

% Rotating CCW direction starting from x-axis
% TO Dos:
%   Add direction configurations
%   Expand to cone-beam projection


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
    if (abs(SourceX-DetectorX)<=abs(SourceY-DetectorY)) % check direction of ray
        for detector_index=1:size(DetectorIndex,2)
            DetectorBoundary1=DetectorIndex(detector_index)-DetectorPixelSize/2;
            DetectorBoundary2=DetectorIndex(detector_index)-DetectorPixelSize/2;
            k1=(SourceX-DetectorBoundary1)/(SourceY-DetectorY);
            intercept1=k1*SourceX-SourceY;
            k2=(SourceX-DetectorBoundary2)/(SourceY-DetectorY); % slope of line between source and detector boundray
            intercept2=k2*SourceX-SourceY;
            detector_value=0;
            for image_row_index=1:ny
                image_col_index1=floor(k1*(Yplane(image_row_index)+dy/2)+intercept1);
                image_col_index2=floor(k2*(Yplane(image_row_index)+dy/2)+intercept2);
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if condition:
                    continue;
                end
                if( image_col_index1==image_col_index2)
                    detector_value=detector_value+ph(image_row_index,image_col_index)*detector_weight;
                    % check order of phantom image
                else
                    detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                        (image_col_index2-DetectorBoundary1)/(DetectorBoundary2-DetectorBoundary1)+...
                        ph(image_row_index,image_col_index2)*(DetectorBoundary2-image_col_index2)/...
                        (DetectorBoundary2-DetectorBoundary1);
                end
            end
        end
    else
    end
end
imagesc(proj);
colormap gray;
toc
