tic;
nx=512;
ny=nx;
ph=phantom(nx);
Source_init=[0,1000]; % Initial source position
Detector_init=[0,-500]; % Initial detector position
Origin=[0,0]; % Rotating center
SAD=sqrt(sum((Source_init-Origin).^2));
SDD=sqrt(sum((Source_init-Detector_init).^2));
DetectorPixelSize=0.5; % Detector pixel spacing
NumberOfDetectorPixels=[750 ,1]; % Number of detector rows and chnnels
PhantomCenter=[0,0]; % Center of phantom
dx=0.5; %phantom pixel spacing
dy=0.5;
nTheta=1;
StartAngle=0;
EndAngle=2*pi;

Xplane=(PhantomCenter(1)-size(ph,1)/2+(0:nx))*dx; % pixel boundaries of image
Yplane=(PhantomCenter(2)-size(ph,2)/2+(0:ny))*dy;
Xplane=Xplane-dx/2;
Yplane=Yplane-dy/2;
theta=linspace(StartAngle,EndAngle,nTheta+1);
theta=theta(1:end-1);
proj=zeros(NumberOfDetectorPixels(1),nTheta);

% Rotating CCW direction starting from x-axis
% TO Dos:
%   Add direction configurations
%   Expand to cone-beam projection

tmp=zeros(size(ny));
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
%         for detector_index=134:134
            DetectorBoundary1=DetectorIndex(1,detector_index)-DetectorPixelSize/2;
            DetectorBoundary2=DetectorIndex(1,detector_index)+DetectorPixelSize/2;
            k1=(SourceX-DetectorBoundary1)/(SourceY-DetectorY);
            intercept1=-k1*SourceY+SourceX;
            k2=(SourceX-DetectorBoundary2)/(SourceY-DetectorY); % slope of line between source and detector boundray
            intercept2=-k2*SourceY+SourceX;
            detector_value=0;
            for image_row_index=ny:-1:1
                coord1=k1*Yplane(image_row_index)+intercept1;
                coord2=k2*Yplane(image_row_index)+intercept2;
%                 tmp(image_row_index)=coord2-coord1;
                fprintf('%.5f\n',coord2-coord1)
                if(max(coord1,coord2)<Xplane(1) || min(coord1,coord2)>Xplane(end))
                    continue;
                end
                image_col_index1=floor((coord1-Xplane(1)+dx)/dx);
                image_col_index2=floor((coord2-Xplane(1)+dx)/dx);
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if( image_col_index1==image_col_index2)
                    detector_value=detector_value+ph(image_row_index,image_col_index1);%/(coord2-coord1);
                    tmp(image_row_index)=ph(image_row_index,image_col_index1);
                    % check order of phantom image
                else
                    if(coord1<Xplane(1))
                        detector_value=detector_value+...
                            ph(image_row_index,image_col_index2)*(coord2-Xplane(image_col_index2))/...
                            (coord2-coord1);
                    elseif(coord2>Xplane(end))
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            (Xplane(image_col_index1)-coord1)/(coord2-coord1);
                    elseif(image_col_index2-image_col_index1>1)
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            (Xplane(image_col_index1+1)-coord1)/(coord2-coord1);
                        for pixels=image_col_index1+1:image_col_index2-1
                            detector_value=detector_value+ph(image_row_index,pixels)...
                                *(Xplane(image_col_index2)-Xplane(image_col_index1+1))/(coord2-coord1);
                        end
                        detector_value=detector_value+ph(image_row_index,image_col_index2)*...
                            (coord2-Xplane(image_col_index2))/(coord2-coord1);
                        
                    else
                        tmp(image_row_index)=ph(image_row_index,image_col_index1)*...
                            (Xplane(image_col_index2)-coord1)/(coord2-coord1)+...
                            ph(image_row_index,image_col_index2)*(coord2-Xplane(image_col_index2))/...
                            (coord2-coord1);
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            (Xplane(image_col_index2)-coord1)/(coord2-coord1)+...
                            ph(image_row_index,image_col_index2)*(coord2-Xplane(image_col_index2))/...
                            (coord2-coord1);
                    end
%                     for pixel_index=image_col_index1:image_col_index2
%                         detector_value=detector_value+ph(image_row_index,pixel_index)*...
%                             (Xplane(pixel_index)-DetectorBoundary1)
                    
                end
            end
            proj(detector_index,angle_index)=detector_value;
        end
    else
        % if projection is done on 
        for detector_index=1:size(DetectorIndex,2)
            DetectorBoundary1=DetectorIndex(2,detector_index)-DetectorPixelSize/2;
            DetectorBoundary2=DetectorIndex(2,detector_index)+DetectorPixelSize/2;
            k1=(SourceX-DetectorBoundary1)/(SourceY-DetectorY);
            intercept1=-k1*SourceX+SourceY;
            k2=(SourceX-DetectorBoundary2)/(SourceY-DetectorY); % slope of line between source and detector boundray
            intercept2=-k2*SourceX+SourceY;
            detector_value=0;
            for image_col_index=1:nx
                image_row_index1=floor((k1*Xplane(image_col_index)+intercept1-Yplane(1)+dy)/dy);
                image_row_index2=floor((k2*Xplane(image_col_index)+intercept2-Yplane(1)+dy)/dy);
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if (image_row_index1<1 || image_row_index2>=size(Yplane,2) )
                    continue;
                end
                if( image_col_index1==image_col_index2)
                    detector_value=detector_value+ph(image_row_index1,image_col_index)*DetectorPixelSize;
                    % check order of phantom image
                else
                    detector_value=detector_value+ph(image_row_index1,image_col_index)*...
                        (Xplane(image_col_index2)-DetectorBoundary1)/DetectorPixelSize+...
                        ph(image_row_index2,image_col_index)*(DetectorBoundary2-Xplane(image_col_index2))/...
                        DetectorPixelSize;
                end
            end
        end
    end
    
end
plot(proj);
colormap gray;
toc
