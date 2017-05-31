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
NumberOfDetectorPixels=[1024 ,1]; % Number of detector rows and chnnels
PhantomCenter=[0,0]; % Center of phantom
dx=0.5; %phantom pixel spacing
dy=0.5;
nTheta=180;
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
weight_map=zeros(size(ph,1),size(ph,2),nTheta);
for angle_index=1:nTheta
    
    SourceX=-SAD*sin(theta(angle_index)); % source coordinate
    SourceY=SAD*cos(theta(angle_index));
    DetectorX=(SDD-SAD)*sin(theta(angle_index));  % center of detector coordinate
    DetectorY=-(SDD-SAD)*cos(theta(angle_index));
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
%     DetectorIndex=DetectorIndex(:,1:end-1); % The index pointing center of detector pixels
    for detector_index=1:size(DetectorIndex,2)-1
        
        if(abs(SourceX-DetectorIndex(1,detector_index))<=abs(SourceY-DetectorIndex(2,detector_index)))
%             DetectorBoundary1=[DetectorIndex(1,detector_index)-cos(theta(angle_index))*...
%                 DetectorPixelSize/2,DetectorIndex(2,detector_index)-sin(theta(angle_index))*...
%                 DetectorPixelSize/2];
%             DetectorBoundary2=[DetectorIndex(1,detector_index)+cos(theta(angle_index))*...
%                 DetectorPixelSize/2,DetectorIndex(2,detector_index)+sin(theta(angle_index))*...
%                 DetectorPixelSize/2];
            DetectorBoundary1=DetectorIndex(:,detector_index);
            DetectorBoundary2=DetectorIndex(:,detector_index+1);
            k1=(SourceX-DetectorBoundary1(1))/(SourceY-DetectorBoundary1(2));
            intercept1=-k1*SourceY+SourceX;
            k2=(SourceX-DetectorBoundary2(1))/(SourceY-DetectorBoundary2(2)); % slope of line between source and detector boundray
            intercept2=-k2*SourceY+SourceX;
            detector_value=0;
            for image_row_index=1:ny
                coord1=k1*Yplane(image_row_index)+intercept1; % x coordinate of detector pixel onto image pixel
                coord2=k2*Yplane(image_row_index)+intercept2;
                if(max(coord1,coord2)<Xplane(1) || min(coord1,coord2)>=Xplane(end))
                    continue;
                end
                image_col_index1=floor((coord1-Xplane(1)+dx)/dx);
                image_col_index2=floor((coord2-Xplane(1)+dx)/dx);
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if( image_col_index1==image_col_index2)
                    detector_value=detector_value+ph(image_row_index,image_col_index1);%/(coord2-coord1);
                    weight_map(image_row_index,image_col_index1,angle_index)=weight_map(image_row_index,image_col_index1,angle_index)+1;
                    tmp(image_row_index)=ph(image_row_index,image_col_index1);
                    % check order of phantom image
                else
                    if(min(coord1,coord2)<Xplane(1))
                        if(image_col_index1>0 && Xplane(image_col_index1)>min(coord1,coord2) && Xplane(image_col_index1)<=max(coord1,coord2))
                            pixel=image_col_index1;
                        else
                            pixel=image_col_index2;
                        end
                        detector_value=detector_value+...
                            ph(image_row_index,pixel)*(coord2-Xplane(pixel))/...
                            (coord2-coord1);
                        weight_map(image_row_index,pixel,angle_index)=weight_map(image_row_index,pixel,angle_index)+...
                            (coord2-Xplane(pixel))/(coord2-coord1);
                    elseif(max(coord1,coord2)>Xplane(end))
                        if(image_col_index1>length(Xplane) || image_col_index1<1)
                            pixel=image_col_index2+1;
                        elseif(image_col_index2>length(Xplane) || image_col_index2<1)
                            pixel=image_col_index1+1;
                        elseif(Xplane(image_col_index1)<Xplane(image_col_index2))
                            pixel=image_col_index1+1;
                        else
                            pixel=image_col_index2+1;
                        end
                        detector_value=detector_value+ph(image_row_index,pixel-1)*...
                            (Xplane(pixel)-coord1)/(coord2-coord1);
                        weight_map(image_row_index,pixel-1,angle_index)=weight_map(image_row_index,pixel-1,angle_index)+...
                            (Xplane(pixel)-coord1)/(coord2-coord1);
                    elseif(abs(image_col_index2-image_col_index1)>1)
                        min_plane=min(Xplane(image_col_index1),Xplane(image_col_index2));
                        max_plane=max(Xplane(image_col_index1),Xplane(image_col_index2));
                        if(min_plane==Xplane(image_col_index1))
                            min_plane_index=image_col_index1;
                            max_plane_index=image_col_index2;
                        else
                            min_plane_index=image_col_index2;
                            max_plane_index=image_col_index1;
                        end
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            ((min_plane+dx)-coord1)/(coord2-coord1);
                        weight_map(image_row_index,image_col_index1,angle_index)=weight_map(image_row_index,image_col_index1,angle_index)+...
                            ((min_plane+dx)-coord1)/(coord2-coord1);
                        for pixels=min_plane_index+1:max_plane_index-1
                            detector_value=detector_value+ph(image_row_index,pixels)...
                                *(dx)/(coord2-coord1);
                            weight_map(image_row_index,pixels,angle_index)=weight_map(image_row_index,pixels,angle_index)+...
                            (Xplane(pixels+1)-Xplane(pixels))/(coord2-coord1);
                        end
                        detector_value=detector_value+ph(image_row_index,image_col_index2)*...
                            (coord2-max_plane)/(coord2-coord1);
                         weight_map(image_row_index,image_col_index2,angle_index)=weight_map(image_row_index,image_col_index2,angle_index)+...
                            (coord2-max_plane)/(coord2-coord1);
                        
                    else
                        max_plane=max(Xplane(image_col_index1),Xplane(image_col_index2));
                        tmp(image_row_index)=ph(image_row_index,image_col_index1)*...
                            (max_plane-coord1)/(coord2-coord1)+...
                            ph(image_row_index,image_col_index2)*(coord2-max_plane)/...
                            (coord2-coord1);
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            (max_plane-coord1)/(coord2-coord1)+...
                            ph(image_row_index,image_col_index2)*(coord2-max_plane)/...
                            (coord2-coord1);
                        weight_map(image_row_index,image_col_index1,angle_index)=weight_map(image_row_index,image_col_index1,angle_index)+...
                            (max_plane-coord1)/(coord2-coord1);
                        weight_map(image_row_index,image_col_index2,angle_index)=weight_map(image_row_index,image_col_index2,angle_index)+...
                            (coord2-max_plane)/(coord2-coord1);
                    end
                end
            end
            proj(detector_index,angle_index)=detector_value;
        else
%             DetectorBoundary1=[DetectorIndex(1,detector_index)-cos(theta(angle_index))*...
%                 DetectorPixelSize/2,DetectorIndex(2,detector_index)-sin(theta(angle_index))*...
%                 DetectorPixelSize/2];
%             DetectorBoundary2=[DetectorIndex(1,detector_index)+cos(theta(angle_index))*...
%                 DetectorPixelSize/2,DetectorIndex(2,detector_index)+sin(theta(angle_index))*...
%                 DetectorPixelSize/2];
            DetectorBoundary1=DetectorIndex(:,detector_index);
            DetectorBoundary2=DetectorIndex(:,detector_index+1);
            k1=(SourceY-DetectorBoundary1(2))/(SourceX-DetectorBoundary1(1));
            intercept1=-k1*SourceX+SourceY;
            k2=(SourceY-DetectorBoundary2(2))/(SourceX-DetectorBoundary2(1)); % slope of line between source and detector boundray
            intercept2=-k2*SourceX+SourceY;
            detector_value=0;
            for image_col_index=1:nx
                coord1=k1*Xplane(image_col_index)+intercept1; % y coordinate of detector pixel onto image pixel
                coord2=k2*Xplane(image_col_index)+intercept2;
                if(max(coord1,coord2)<Yplane(1) || min(coord1,coord2)>Yplane(end))
                    continue;
                end
                image_row_index1=floor((coord1-Yplane(1)+dy)/dy);
                image_row_index2=floor((coord2-Yplane(1)+dy)/dy);
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if( image_row_index1==image_row_index2)
                    detector_value=detector_value+ph(image_row_index1,image_col_index);%/(coord2-coord1);
                    weight_map(image_row_index1,image_col_index,angle_index)=weight_map(image_row_index1,image_col_index,angle_index)+1;
                    tmp(image_col_index)=ph(image_row_index1,image_col_index);
                    % check order of phantom image
                else
                    if(min(coord1,coord2)<Yplane(1))
                        if(image_row_index1<1 || image_row_index1>length(Yplane))
                            pixel=image_row_index2;
                        elseif(image_row_index2<1|| image_row_index2>length(Yplane))
                            pixel=image_row_index1;
                        elseif(Yplane(image_row_index1)<Yplane(image_row_index2))
                            pixel=image_row_index2;
                        else
                            pixel=image_row_index1;
                        end
                        detector_value=detector_value+...
                            ph(pixel,image_col_index)*(coord2-Yplane(pixel))/...
                            (coord2-coord1);
                        weight_map(pixel,image_col_index,angle_index)=weight_map(pixel,image_col_index,angle_index)+...
                            (coord2-Yplane(pixel))/(coord2-coord1);
                    elseif(max(coord1,coord2)>Yplane(end))
                        if(image_row_index1<1 || image_row_index1>length(Yplane))
                            pixel=image_row_index2+1;
                        elseif(image_row_index2<1 || image_row_index2>length(Yplane))
                            pixel=image_row_index1+1;
                        elseif(Yplane(image_row_index1)<Yplane(image_row_index2))
                            pixel=image_row_index1+1;
                        else
                            pixel=image_row_index2+1;
                        end
                        detector_value=detector_value+ph(pixel-1,image_col_index)*...
                            (Yplane(pixel)-coord1)/(coord2-coord1);
                        weight_map(pixel-1,image_col_index,angle_index)=weight_map(pixel-1,image_col_index,angle_index)+...
                            (Yplane(pixel)-coord1)/(coord2-coord1);
                    elseif(abs(image_row_index2-image_row_index1)>1)
                        min_plane=min(Yplane(image_row_index1),Yplane(image_row_index2));
                        max_plane=max(Yplane(image_row_index1),Yplane(image_row_index2));
                        if(min_plane==Yplane(image_row_index1))
                            min_plane_index=image_row_index1;
                            max_plane_index=image_row_index2;
                        else
                            min_plane_index=image_row_index2;
                            max_plane_index=image_row_index1;
                        end
                        detector_value=detector_value+ph(min_plane_index,image_col_index)*...
                            ((min_plane+dx)-coord1)/(coord2-coord1);
                        weight_map(min_plane_index,image_col_index,angle_index)=weight_map(min_plane_index,image_col_index,angle_index)+...
                            ((min_plane+dx)-coord1)/(coord2-coord1);
                        for pixels=min_plane_index+1:max_plane_index-1
                            detector_value=detector_value+ph(pixels,image_col_index)...
                                *(dx)/(coord2-coord1);
                            weight_map(pixels,image_col_index,angle_index)=weight_map(pixels,image_col_index,angle_index)+...
                            (Yplane(pixels+1)-Yplane(pixels))/(coord2-coord1);
                        end
                        detector_value=detector_value+ph(max_plane_index,image_col_index)*...
                            (coord2-max_plane)/(coord2-coord1);
                        weight_map(max_plane_index,image_col_index,angle_index)=weight_map(max_plane_index,image_col_index,angle_index)+...
                            (coord2-max_plane)/(coord2-coord1);
                        
                    else
                        max_plane=max(Yplane(image_row_index1),Yplane(image_row_index2));
                        tmp(image_col_index)=ph(image_row_index1,image_col_index)*...
                            (max_plane-coord1)/(coord2-coord1)+...
                            ph(image_row_index2,image_col_index)*(coord2-...
                            max_plane)/(coord2-coord1);
                        detector_value=detector_value+ph(image_row_index1,image_col_index)*...
                            (max_plane-coord1)/(coord2-coord1)+...
                            ph(image_row_index2,image_col_index)*(coord2-...
                            max_plane)/(coord2-coord1);
                        weight_map(image_row_index1,image_col_index,angle_index)=weight_map(image_row_index1,image_col_index,angle_index)+...
                            (max_plane-coord1)/(coord2-coord1);
                        weight_map(image_row_index2,image_col_index,angle_index)=weight_map(image_row_index2,image_col_index,angle_index)+...
                            (coord2-max_plane)/(coord2-coord1);
                    end
                end
            end
            proj(detector_index,angle_index)=detector_value;
        end
    end
end
%     fprintf('%d %f\n',angle_index,max(weight_map(:)));
% plot(proj);
imagesc(proj);
colormap gray;
toc
