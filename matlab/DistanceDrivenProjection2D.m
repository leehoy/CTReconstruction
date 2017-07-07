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

tol_min=1e-7;

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

% Normalization for each pixel and ray is required
% maximum weight value at angle 50 is smaller than 23
weight_map=zeros(size(ph,1),size(ph,2),nTheta);
% ray_angle=zeros(nTheta,NumberOfDetectorPixels(1));

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
    DetectorIndex=DetectorIndex(:,1:end-1); % The index pointing center of detector pixels
    for detector_index=1:size(DetectorIndex,2)
        if(abs(SourceX-DetectorIndex(1,detector_index))<abs(SourceY-DetectorIndex(2,detector_index)))
            DetectorBoundary1=[DetectorIndex(1,detector_index)-cos(theta(angle_index))*...
                DetectorPixelSize/2,DetectorIndex(2,detector_index)-sin(theta(angle_index))*...
                DetectorPixelSize/2];
            DetectorBoundary2=[DetectorIndex(1,detector_index)+cos(theta(angle_index))*...
                DetectorPixelSize/2,DetectorIndex(2,detector_index)+sin(theta(angle_index))*...
                DetectorPixelSize/2];
%             DetectorBoundary1=DetectorIndex(:,detector_index); % this values need to be changed to poiting boundary of detector cell
%             DetectorBoundary2=DetectorIndex(:,detector_index+1);% this values need to be changed to poiting boundary of detector cell
            k1=(SourceX-DetectorBoundary1(1))/(SourceY-DetectorBoundary1(2));
            intercept1=-k1*SourceY+SourceX;
            k2=(SourceX-DetectorBoundary2(1))/(SourceY-DetectorBoundary2(2)); % slope of line between source and detector boundray
            intercept2=-k2*SourceY+SourceX;
            ray_angle=atand(sqrt(sum((DetectorIndex(:,detector_index)-[DetectorX;DetectorY]).^2))/SDD);
            ray_normalization=cosd(ray_angle);
%             ray_normalization=abs(cos(atan((SourceX-DetectorIndex(1,detector_index))/(SourceY-DetectorIndex(2,detector_index)))));
%             ray_normalization=abs(cos(atan2((SourceY-DetectorIndex(2,detector_index)),(SourceX-DetectorIndex(1,detector_index)))));
            detector_value=0;
            for image_row_index=1:ny
                coord1=k1*(Yplane(image_row_index)+dy/2)+intercept1; % x coordinate of detector pixel onto image pixel
                coord2=k2*(Yplane(image_row_index)+dy/2)+intercept2;
                if(max(coord1,coord2)<Xplane(1) || min(coord1,coord2)>=Xplane(end))
                    continue;
                end
                image_col_index1=floor((coord1-Xplane(1)+dx)/dx);
                image_col_index2=floor((coord2-Xplane(1)+dx)/dx);
%                 image_col_index1=floor(coord1)+1;
%                 image_col_index2=floor(coord2)+1;
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if(max(image_col_index1,image_col_index2)<=1 || min(image_col_index1,image_col_index2)>=nx)
                    continue;
                end
                if( image_col_index1==image_col_index2 )
%                   ray passing a single voxel
                    detector_value=detector_value+ph(image_row_index,image_col_index1);%/(coord2-coord1);
                    weight_map(image_row_index,image_col_index1,angle_index)=...
                        weight_map(image_row_index,image_col_index1,angle_index)+1;
                    % check order of phantom image
                else
                    if(min(coord1,coord2)<=Xplane(1))
%                       One of the ray not passing the phantom
%                       left boundary
                        if(coord1<=Xplane(1))
                            pixel=image_col_index2;
                            weight=coord2-Xplane(pixel);
                        else
                            pixel=image_col_index1;
                            weight=coord1-Xplane(pixel);
                        end
                        if(abs(weight)<tol_min)
                            weight=0;
                        end
                        weight=abs(weight);
                        detector_value=detector_value+...
                            ph(image_row_index,pixel)*weight/...
                            abs(coord2-coord1);
                        weight_map(image_row_index,pixel,angle_index)=...
                            weight_map(image_row_index,pixel,angle_index)+...
                            weight/abs(coord2-coord1);
                    elseif(max(coord1,coord2)>=Xplane(end))
%                       One of the ray not passing the phantom
%                       right boundary
                        if(coord2>=Xplane(end))
                            pixel=image_col_index1+1;
                            weight=Xplane(pixel)-coord1;
                        else
                            pixel=image_col_index2+1;
                            weight=Xplane(pixel)-coord2;
                        end
                        if(abs(weight)<tol_min)
                            weight=0;
                        end
                        weight=abs(weight);
                        detector_value=detector_value+ph(image_row_index,pixel-1)*...
                            weight/abs(coord2-coord1);
                        weight_map(image_row_index,pixel-1,angle_index)=...
                            weight_map(image_row_index,pixel-1,angle_index)+...
                            weight/abs(coord2-coord1);
                    elseif(abs(image_col_index2-image_col_index1)>1)
%                       Both ray passing through the phantom, they are
%                       separated more than 2 voxels
                        min_plane=min(Xplane(image_col_index1),Xplane(image_col_index2));
                        max_plane=max(Xplane(image_col_index1),Xplane(image_col_index2));
                        min_coord=min(coord1,coord2);
                        max_coord=max(coord1,coord2);
                        if(min_plane==Xplane(image_col_index1))
                            min_plane_index=image_col_index1;
                            max_plane_index=image_col_index2;
                        else
                            min_plane_index=image_col_index2;
                            max_plane_index=image_col_index1;
                        end
                        weight_min=(min_plane+dx)-min_coord;
                        if(abs(weight_min)<tol_min)
                            weight_min=0;
                        end
                        weight_min=abs(weight_min);
%                         coord1 is not always bigger than coord2, but it
%                         doesn't matter
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            weight_min/abs(coord2-coord1);
                        weight_map(image_row_index,image_col_index1,angle_index)=...
                            weight_map(image_row_index,image_col_index1,angle_index)+...
                            weight_min/abs(coord2-coord1);
                        for pixels=min_plane_index+1:max_plane_index-1
                            detector_value=detector_value+ph(image_row_index,pixels)...
                                *dx/abs(coord2-coord1);
                            weight_map(image_row_index,pixels,angle_index)=...
                                weight_map(image_row_index,pixels,angle_index)+...
                                dx/abs(coord2-coord1);
                        end
                        weight_max=max_coord-max_plane;
                        if(abs(weight_max)<tol_min)
                            weight_max=0;
                        end
                        weight_max=abs(weight_max);
                        detector_value=detector_value+ph(image_row_index,image_col_index2)*...
                            weight_max/abs(coord2-coord1);
                         weight_map(image_row_index,image_col_index2,angle_index)=...
                             weight_map(image_row_index,image_col_index2,angle_index)+...
                             weight_max/abs(coord2-coord1);
                    else
                        max_plane=max(Xplane(image_col_index1),Xplane(image_col_index2));
                        weight1=max_plane-coord1;
                        weight2=coord2-max_plane;
                        if(abs(weight1)<tol_min)
                            weight1=0;
                        end
                        if(abs(weight2)<tol_min)
                            weight2=0;
                        end
                        weight1=abs(weight1);
                        weight2=abs(weight2);
                        detector_value=detector_value+ph(image_row_index,image_col_index1)*...
                            weight1/(coord2-coord1)+...
                            ph(image_row_index,image_col_index2)*weight2/...
                            (coord2-coord1);
                        weight_map(image_row_index,image_col_index1,angle_index)=...
                            weight_map(image_row_index,image_col_index1,angle_index)+...
                            weight1/(coord2-coord1);
                        weight_map(image_row_index,image_col_index2,angle_index)=...
                            weight_map(image_row_index,image_col_index2,angle_index)+...
                            weight2/(coord2-coord1);
                    end
                end
%                 detector_value=detector_value/ray_normalization;
%                 figure(1);
%                 plot(ray_normalization);
            end
            proj(detector_index,angle_index)=detector_value*DetectorPixelSize/ray_normalization;
        else
            DetectorBoundary1=[DetectorIndex(1,detector_index)-cos(theta(angle_index))*...
                DetectorPixelSize/2,DetectorIndex(2,detector_index)-sin(theta(angle_index))*...
                DetectorPixelSize/2];
            DetectorBoundary2=[DetectorIndex(1,detector_index)+cos(theta(angle_index))*...
                DetectorPixelSize/2,DetectorIndex(2,detector_index)+sin(theta(angle_index))*...
                DetectorPixelSize/2];
            k1=(SourceY-DetectorBoundary1(2))/(SourceX-DetectorBoundary1(1));
            intercept1=-k1*SourceX+SourceY;
            k2=(SourceY-DetectorBoundary2(2))/(SourceX-DetectorBoundary2(1)); % slope of line between source and detector boundray
            intercept2=-k2*SourceX+SourceY;
            ray_angle=atand(sqrt(sum((DetectorIndex(:,detector_index)-[DetectorX;DetectorY]).^2))/SDD);
            ray_normalization=cosd(ray_angle);
            detector_value=0;
            for image_col_index=1:nx
                coord1=k1*(Xplane(image_col_index)+dx/2)+intercept1; % y coordinate of detector pixel onto image pixel
                coord2=k2*(Xplane(image_col_index)+dx/2)+intercept2;
                if(max(coord1,coord2)<Yplane(1) || min(coord1,coord2)>Yplane(end))
                    continue;
                end
                image_row_index1=floor((coord1-Yplane(1)+dy)/dy);
                image_row_index2=floor((coord2-Yplane(1)+dy)/dy);
                % image_col_indexes are real cooordinate, not index
                % how to check the line intersection is out of the phantom
                % or not?
                if(max(image_row_index1,image_row_index2)<=1 || min(image_row_index1,image_row_index2)>=ny)
                    continue;
                end
                if( image_row_index1==image_row_index2)
                    detector_value=detector_value+ph(image_row_index1,image_col_index);%/(coord2-coord1);
                    weight_map(image_row_index1,image_col_index,angle_index)=...
                        weight_map(image_row_index1,image_col_index,angle_index)+1;
                    % check order of phantom image
                else
                    if(min(coord1,coord2)<=Yplane(1))
                        if(coord1<=Yplane(1))
                            pixel=image_row_index2;
                            weight=coord2-Yplane(pixel);
                        else
                            pixel=image_row_index1;
                            weight=coord1-Yplane(pixel);
                        end
                        if(abs(weight)<tol_min)
                            weight=0;
                        end
                        weight=abs(weight);
                        detector_value=detector_value+...
                            ph(pixel,image_col_index)*weight/abs(coord2-coord1);
                        weight_map(pixel,image_col_index,angle_index)=...
                            weight_map(pixel,image_col_index,angle_index)+...
                            weight/abs(coord2-coord1);
                    elseif(max(coord1,coord2)>=Yplane(end))
                        if(coord2>=Yplane(end))
                            pixel=image_row_index1+1;
                            weight=Yplane(pixel)-coord1;
                        else
                            pixel=image_row_index2+1;
                            weight=Yplane(pixel)-coord2;
                        end
                        if(abs(weight)<tol_min)
                            weight=0;
                        end
                        weight=abs(weight);
                        detector_value=detector_value+ph(pixel-1,image_col_index)*...
                            weight/abs(coord2-coord1);
                        weight_map(pixel-1,image_col_index,angle_index)=...
                            weight_map(pixel-1,image_col_index,angle_index)+...
                            weight/abs(coord2-coord1);
                    elseif(abs(image_row_index2-image_row_index1)>1)
                        min_plane=min(Yplane(image_row_index1),Yplane(image_row_index2));
                        max_plane=max(Yplane(image_row_index1),Yplane(image_row_index2));
                        min_coord=min(coord1,coord2);
                        max_coord=max(coord1,coord2);
                        if(min_plane==Yplane(image_row_index1))
                            min_plane_index=image_row_index1;
                            max_plane_index=image_row_index2;
                        else
                            min_plane_index=image_row_index2;
                            max_plane_index=image_row_index1;
                        end
                        weight_min=(min_plane+dy)-min_coord;
                        weight_min=abs(weight_min);
                        detector_value=detector_value+ph(min_plane_index,image_col_index)*...
                            weight_min/abs(coord2-coord1);
                        weight_map(min_plane_index,image_col_index,angle_index)=...
                            weight_map(min_plane_index,image_col_index,angle_index)+...
                            weight_min/abs(coord2-coord1);
                        for pixels=min_plane_index+1:max_plane_index-1
                            detector_value=detector_value+ph(pixels,image_col_index)...
                                *dy/abs(coord2-coord1);
                            weight_map(pixels,image_col_index,angle_index)=...
                                weight_map(pixels,image_col_index,angle_index)+...
                                dy/abs(coord2-coord1);
                        end
                        weight_max=max_coord-max_plane;
                        weight_max=abs(weight_max);
                        if(abs(weight_max)<tol_min)
                            weight_max=0;
                        end
                        detector_value=detector_value+ph(max_plane_index,image_col_index)*...
                            weight_max/abs(coord2-coord1);
                        weight_map(max_plane_index,image_col_index,angle_index)=...
                            weight_map(max_plane_index,image_col_index,angle_index)+...
                            weight_max/abs(coord2-coord1);
                        
                    else
                        max_plane=max(Yplane(image_row_index1),Yplane(image_row_index2));
                        weight1=max_plane-coord1;
                        weight2=coord2-max_plane;
                        if(abs(weight1)<tol_min)
                            weight1=0;
                        end
                        if(abs(weight2)<tol_min)
                            weight2=0;
                        end
                        weight1=abs(weight1);
                        weight2=abs(weight2);
                        detector_value=detector_value+ph(image_row_index1,image_col_index)*...
                            weight1/(coord2-coord1)+...
                            ph(image_row_index2,image_col_index)*weight2/(coord2-coord1);
                        weight_map(image_row_index1,image_col_index,angle_index)=...
                            weight_map(image_row_index1,image_col_index,angle_index)+...
                            weight1/(coord2-coord1);
                        weight_map(image_row_index2,image_col_index,angle_index)=...
                            weight_map(image_row_index2,image_col_index,angle_index)+...
                            weight2/(coord2-coord1);
                    end
                end
%                 detector_value=detector_value/ray_normalization;
            end
            proj(detector_index,angle_index)=detector_value*DetectorPixelSize/ray_normalization;
        end
    end
% %     fprintf('%d %f\n',angle_index,max(max(weight_map(:,:,angle_index))));
end
    
% plot(proj);
imagesc(proj);
colormap gray;
toc
