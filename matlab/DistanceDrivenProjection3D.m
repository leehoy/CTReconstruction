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
DetectorPixelSize=[0.5,0.5]; % Detector pixel spacing
NumberOfDetectorPixels=[512 ,384]; % Number of detector rows and chnnels
PhantomCenter=[0,0,0]; % Center of phantom
PhantomPixelSpacingX=0.5;
PhantomPixelSpacingY=0.5;
PhantomPixelSpacingZ=0.5;
nTheta=90;
StartAngle=0;
EndAngle=2*pi;

tol_min=1e-7;
dx=PhantomPixelSpacingX;
dy=-PhantomPixelSpacingY;
dz=-PhantomPixelSpacingZ;

Xplane=(PhantomCenter(1)-size(ph,1)/2+(0:nx))*dx; % pixel boundaries of image
Yplane=(PhantomCenter(2)-size(ph,2)/2+(0:ny))*dy;
Zplane=(PhantomCenter(3)-size(ph,3)/2+(0:nz))*dz;
Xplane=Xplane-dx/2;
Yplane=Yplane-dy/2;
Zplane=Zplane-dz/2;
DetectorPixelSizeH=DetectorPixelSize(1);
DetectorPixelSizeV=DetectorPixelSize(2);
theta=linspace(StartAngle,EndAngle,nTheta+1);
theta=theta(1:end-1);
proj=zeros(NumberOfDetectorPixels(1),NumberOfDetectorPixels(2),nTheta);

% Rotating CCW direction starting from x-axis
% TO Dos:
%   Add direction configurations
%   Expand to cone-beam projection
%   Compatitable with arbitrary trajectories

% Normalization for each pixel and ray is required
% maximum weight value at angle 50 is smaller than 23
weight_map=zeros(size(ph,1),size(ph,2),nTheta);
% ray_angle=zeros(nTheta,NumberOfDetectorPixels(1));

% Total number of beam decreases

for angle_index=1:nTheta
    SourceX=-SAD*sin(theta(angle_index)); % source coordinate
    SourceY=SAD*cos(theta(angle_index));
    SourceZ=0;
    DetectorX=(SDD-SAD)*sin(theta(angle_index));  % center of detector coordinate
    DetectorY=-(SDD-SAD)*cos(theta(angle_index));
    DetectorZ=0;
    DetectorLengthH=(floor(-NumberOfDetectorPixels(1)/2):floor(NumberOfDetectorPixels(1)/2))*DetectorPixelSizeH; %horizontal detector length
    DetectorLengthV=(floor(-NumberOfDetectorPixels(2)/2):floor(NumberOfDetectorPixels(2)/2))*DetectorPixelSizeV; %horizontal detector length
    if(abs(tan(theta(angle_index)))<=1e-6) % detector is parallel to x-axis
        DetectorIndex=[DetectorX+DetectorLengthH; repmat(DetectorY,1,size(DetectorLengthH,2))];
        DetectorIndexZ=DetectorZ-DetectorLengthV;
    elseif(tan(theta(angle_index))>=1e6) % detector is parallel to y-axis
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
%     weight=zeros(ny, size(Xplane,2)-1);
    for detector_index_h=1:size(DetectorIndex,2)
        for detector_index_v=1:size(DetectorIndex,3)
            if(abs(SourceX-DetectorIndex(1,detector_index_h,detector_index_v))<=...
                    abs(SourceY-DetectorIndex(2,detector_index_h,detector_index_v)))
                DetectorBoundary1X=[DetectorIndex(1,detector_index_h,detector_index_v)-cos(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(2,detector_index_h,detector_index_v)-sin(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(3,detector_index_h,detector_index_v)];
                DetectorBoundary2X=[DetectorIndex(1,detector_index_h,detector_index_v)+cos(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(2,detector_index_h,detector_index_v)+sin(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(3,detector_index_h,detector_index_v)];
                DetectorBoundary1Z=[DetectorIndex(1,detector_index_h,detector_index_v),...
                    DetectorIndex(2,detector_index_h,detector_index_v),...
                    DetectorIndex(3,detector_index_h,detector_index_v)-DetectorPixelSizeV/2];
                DetectorBoundary2Z=[DetectorIndex(1,detector_index_h,detector_index_v),...
                    DetectorIndex(2,detector_index_h,detector_index_v),...
                    DetectorIndex(3,detector_index_h,detector_index_v)+DetectorPixelSizeV/2];
                k1X=(SourceX-DetectorBoundary1X(1))/(SourceY-DetectorBoundary1X(2));
                intercept1X=-k1X*SourceY+SourceX;
                k2X=(SourceX-DetectorBoundary2X(1))/(SourceY-DetectorBoundary2X(2)); % slope of line between source and detector boundray
                intercept2X=-k2X*SourceY+SourceX;
                k1Z=(SourceZ-DetectorBoundary1Z(3))/(SourceY-DetectorBoundary1Z(2));
                interceptZ1=-k1Z*SourceY+SourceZ;
                k2Z=(SourceZ-DetectorBoundary2Z(3))/(SourceY-DetectorBoundary2Z(2)); % slope of line between source and detector boundray
                interceptZ2=-k2Z*SourceY+SourceZ;
                ray_angle=atand(sqrt(sum((DetectorIndex(:,detector_index_h,detector_index_v)...
                    -[DetectorX;DetectorY;DetectorZ]).^2))/SDD);
                ray_normalization=cosd(ray_angle);
                detector_value=0;
                for image_y_index=1:ny
                    coord1X=k1X*(Yplane(image_y_index)+dy/2)+intercept1X; % x coordinate of detector pixel onto image pixel
                    coord2X=k2X*(Yplane(image_y_index)+dy/2)+intercept2X;
                    coord1Z=k1Z*(Yplane(image_y_index)+dy/2)+interceptZ1; % x coordinate of detector pixel onto image pixel
                    coord2Z=k2Z*(Yplane(image_y_index)+dy/2)+interceptZ2;
%                     fprintf('%f %f %f %f\n',coord1X,coord2X,coord1Z,coord2Z);
                    if(max(coord1X,coord2X)<min(Xplane(:)) || min(coord1X,coord2X)>max(Xplane(:))...
                            ||abs(max(coord1X,coord2X)-min(Xplane(:)))<=tol_min || ...
                            abs(min(coord1X,coord2X)-max(Xplane(:)))<=tol_min ||...
                            max(coord1Z,coord2Z)<min(Zplane(:)) || min(coord1Z,coord2Z)>max(Zplane(:))...
                            ||abs(max(coord1Z,coord2Z)-min(Zplane(:)))<=tol_min || ...
                            abs(min(coord1Z,coord2Z)-max(Zplane(:)))<=tol_min)
                        continue;
                    end
                    intersection_slope1=(SourceX-DetectorIndex(1,detector_index_h,detector_index_v))...
                        /(SourceY-DetectorIndex(2,detector_index_h,detector_index_v));
                    intersection_slope2=(SourceZ-DetectorIndex(3,detector_index_h,detector_index_v))...
                        /(SourceY-DetectorIndex(2,detector_index_h,detector_index_v));
                    intersection_length=abs(dy)/(cos(atan(intersection_slope1))*cos(atan(intersection_slope2)));
                    image_x_index1=floor((coord1X-Xplane(1)+dx)/dx);
                    image_x_index2=floor((coord2X-Xplane(1)+dx)/dx);
                    image_z_index1=floor((coord1Z-Zplane(1)+dz)/dz);
                    image_z_index2=floor((coord2Z-Zplane(1)+dz)/dz);
                    TotalWeights=PixelWeightCalculator(coord1X,coord2X,...
                        Xplane,dx,coord1Z,coord2Z,Zplane,dz);
                    Counter1=1;
                    for ix=min(image_x_index1,image_x_index2):max(image_x_index1,image_x_index2)
                        Counter2=1;
                        if(ix<1 || ix>size(ph,1))
                            continue;
                        end
                        for iz=min(image_z_index1,image_z_index2):max(image_z_index1,image_z_index2)
                            if(iz<1 || iz>size(ph,3))
                                continue;
                            end
                            proj(detector_index_h,detector_index_v,angle_index)=...
                                proj(detector_index_h,detector_index_v,angle_index)+TotalWeights(Counter1,Counter2)*...
                                ph(ix,image_y_index,iz)*intersection_length/ray_normalization;
                            Counter2=Counter2+1;
%                             fprintf('image_y_index: %d, ix: %d , iz: %d\n',image_y_index,ix,iz);
                        end
                        Counter1=Counter1+1;
                    end
                end
            else
                DetectorBoundary1Y=[DetectorIndex(1,detector_index_h,detector_index_v)-cos(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(2,detector_index_h,detector_index_v)-sin(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(3,detector_index_h,detector_index_v)];
                DetectorBoundary2Y=[DetectorIndex(1,detector_index_h,detector_index_v)+cos(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(2,detector_index_h,detector_index_v)+sin(theta(angle_index))*...
                    DetectorPixelSizeH/2,DetectorIndex(3,detector_index_h,detector_index_v)];
                DetectorBoundary1Z=[DetectorIndex(1,detector_index_h,detector_index_v),...
                    DetectorIndex(2,detector_index_h,detector_index_v),...
                    DetectorIndex(3,detector_index_h,detector_index_v)-DetectorPixelSizeV/2];
                DetectorBoundary2Z=[DetectorIndex(1,detector_index_h,detector_index_v),...
                    DetectorIndex(2,detector_index_h,detector_index_v),...
                    DetectorIndex(3,detector_index_h,detector_index_v)+DetectorPixelSizeV/2];
                k1Y=(SourceY-DetectorBoundary1Y(2))/(SourceX-DetectorBoundary1Y(1));
                intercept1Y=-k1Y*SourceX+SourceY;
                k2Y=(SourceY-DetectorBoundary2Y(2))/(SourceX-DetectorBoundary2Y(1)); % slope of line between source and detector boundray
                intercept2Y=-k2Y*SourceX+SourceY;
                k1Z=(SourceZ-DetectorBoundary1Z(3))/(SourceX-DetectorBoundary1Z(1));
                interceptZ1=-k1Z*SourceX+SourceZ;
                k2Z=(SourceZ-DetectorBoundary2Z(3))/(SourceX-DetectorBoundary2Z(1)); % slope of line between source and detector boundray
                interceptZ2=-k2Z*SourceX+SourceZ;
                ray_angle=atand(sqrt(sum((DetectorIndex(:,detector_index_h,detector_index_v)...
                    -[DetectorX;DetectorY;DetectorZ]).^2))/SDD);
                ray_normalization=cosd(ray_angle);
                detector_value=0;
                for image_x_index=1:nx
                    coord1Y=k1Y*(Xplane(image_x_index)+dx/2)+intercept1Y; % x coordinate of detector pixel onto image pixel
                    coord2Y=k2Y*(Xplane(image_x_index)+dx/2)+intercept2Y;
                    coord1Z=k1Z*(Xplane(image_x_index)+dx/2)+interceptZ1; % x coordinate of detector pixel onto image pixel
                    coord2Z=k2Z*(Xplane(image_x_index)+dx/2)+interceptZ2;
%                     fprintf('%f %f %f %f\n',coord1X,coord2X,coord1Z,coord2Z);
                    if(max(coord1Y,coord2Y)<min(Yplane(:)) || min(coord1Y,coord2Y)>max(Yplane(:))...
                            ||abs(max(coord1Y,coord2Y)-min(Yplane(:)))<=tol_min || ...
                            abs(min(coord1Y,coord2Y)-max(Yplane(:)))<=tol_min ||...
                            max(coord1Z,coord2Z)<min(Zplane(:)) || min(coord1Z,coord2Z)>max(Zplane(:))...
                            ||abs(max(coord1Z,coord2Z)-min(Zplane(:)))<=tol_min || ...
                            abs(min(coord1Z,coord2Z)-max(Zplane(:)))<=tol_min)
                        continue;
                    end
                    intersection_slope1=(SourceY-DetectorIndex(2,detector_index_h,detector_index_v))...
                        /(SourceX-DetectorIndex(1,detector_index_h,detector_index_v));
                    intersection_slope2=(SourceZ-DetectorIndex(3,detector_index_h,detector_index_v))...
                        /(SourceX-DetectorIndex(1,detector_index_h,detector_index_v));
                    intersection_length=abs(dy)/(cos(atan(intersection_slope1))*cos(atan(intersection_slope2)));
                    image_y_index1=floor((coord1Y-Yplane(1)+dy)/dy);
                    image_y_index2=floor((coord2Y-Yplane(1)+dy)/dy);
                    image_z_index1=floor((coord1Z-Zplane(1)+dz)/dz);
                    image_z_index2=floor((coord2Z-Zplane(1)+dz)/dz);
                    TotalWeights=PixelWeightCalculator(coord1Y,coord2Y,...
                        Yplane,dy,coord1Z,coord2Z,Zplane,dz);
                    Counter1=1;
                    for iy=min(image_y_index1,image_y_index2):max(image_y_index1,image_y_index2)
                        Counter2=1;
                        if(iy<1 || iy>size(ph,1))
                            continue;
                        end
                        for iz=min(image_z_index1,image_z_index2):max(image_z_index1,image_z_index2)
                            if(iz<1 || iz>size(ph,3))
                                continue;
                            end
                            proj(detector_index_h,detector_index_v,angle_index)=...
                                proj(detector_index_h,detector_index_v,angle_index)+TotalWeights(Counter1,Counter2)*...
                                ph(image_x_index,iy,iz)*intersection_length/ray_normalization;
%                             fprintf('image_x_index: %d, iy: %d , iz: %d\n',image_x_index,iy,iz);
                            Counter2=Counter2+1;
                        end
                        Counter1=Counter1+1;
                    end
                end
            end
        end
    end
end
% close all;
% plot(proj);
% imagesc(proj);
% colormap gray;
toc
function weight=WeightCalculator(coord1,coord2,plane,dp)
    tol_min=1e-7;
    index1=floor((coord1-plane(1)+dp)/dp);
    index2=floor((coord2-plane(1)+dp)/dp);
    weight=cell(0);
    k=1;
    if( index1==index2)
        weight{k}=1;
        k=k+1;
    else
        if(min(coord1,coord2)<min(plane(:)) || abs(min(coord1,coord2)-min(plane(:)))<=tol_min)
        %One of the ray not passing the phantom
        %left boundary
            if(coord1<=min(plane(:)))
                pixel=index2;
                if(plane(pixel)<coord2)
                    weight{k}=(coord2-plane(pixel))/abs(coord2-coord1);
                elseif(plane(pixel)>coord2)
                    weight{k}=(coord2-plane(pixel+1))/abs(coord2-coord1);
                end
            else
                pixel=index1;
                if(plane(pixel)<coord1)
                    weight{k}=(coord1-plane(pixel))/abs(coord2-coord1);
                elseif(plane(pixel)>coord1)
                    weight{k}=(coord1-plane(pixel+1))/abs(coord2-coord1);
                end
            end
            if(abs(weight{k})<tol_min)
                weight{k}=0;
            end
            k=k+1;
            min_plane_arg=find(plane==min(plane(:)));
            if(min_plane_arg>pixel)
                for p=pixel+1:min_plane_arg-1
                    weight{k}=dp/(coord2-coord1);
                    k=k+1;
                end
            else
                for p=min_plane_arg:pixel-1
                    weight{k}=dp/(coord2-coord1);
                    k=k+1;
                end
            end
        elseif(max(coord1,coord2)>max(plane(:)) || abs(max(coord1,coord2)-max(plane(:)))<=tol_min)
            %One of the ray not passing the phantom
            %right boundary
            if(coord2>=max(plane(:)))
                pixel=index1;
                if(plane(pixel)<coord1)
                    weight{k}=(plane(pixel+1)-coord1)/abs(coord2-coord1);
                elseif(plane(pixel)>=coord1)
                    weight{k}=(plane(pixel)-coord1)/abs(coord2-coord1);
                end
                
            else
                pixel=index2;
                if(plane(pixel)<coord2)
                    weight{k}=(plane(pixel+1)-coord2)/abs(coord2-coord1);
                elseif(plane(pixel)>=coord2)
                    weight{k}=(plane(pixel)-coord2)/abs(coord2-coord1);
                end
            end
            if(abs(weight{k})<tol_min)
                weight=0;
            end
            k=k+1;
            max_plane_arg=find(plane==max(plane(:)));
            if(max_plane_arg>pixel)
                for p=pixel+1:max_plane_arg-1
                    weight{k}=dp/(coord2-coord1);
                    k=k+1;
                end
            else
                for p=max_plane_arg:pixel-1
                    weight{k}=dp/(coord2-coord1);
                    k=k+1;
                end
            end
            
        elseif(abs(index2-index1)>1)
            %Both ray passing through the phantom, they are
            %separated more than 2 voxels
%             min_plane=min(plane(index1),plane(index2));
%             max_plane=max(plane(index1),plane(index2));
            if((plane(index1)<max(coord1,coord2) || abs(plane(index1)-max(coord1,coord2))<tol_min)...
                    && (plane(index1)>min(coord1,coord2) ||abs(plane(index1)-min(coord1,coord2))<tol_min))
                max_plane=plane(index1);
                min_plane=plane(index2);
                max_plane_index=index1;
                min_plane_index=index2;
                min_coord=coord2;
                max_coord=coord1;
            else
                max_plane=plane(index2);
                min_plane=plane(index1);
                max_plane_index=index2;
                min_plane_index=index1;
                min_coord=coord1;
                max_coord=coord2;
            end
%             min_coord=min(coord1,coord2);
%             max_coord=max(coord1,coord2);
%             if(min_plane==plane(index1))
%                 min_plane_index=index1;
%                 max_plane_index=index2;
%             else
%                 min_plane_index=index2;
%                 max_plane_index=index1;
%             end
%             assert(min_coord<max_coord);
%             assert(min_plane<max_plane);
            assert((max_plane>min(coord1,coord2) && max_plane<max(coord1,coord2) )||( abs(max_plane-min(coord1,coord2))<tol_min || abs(max_plane-max(coord1,coord2))<tol_min))
            weight{k}=((min_plane+dp)-min_coord)/(coord2-coord1);
            if(abs(weight{k})<tol_min)
                weight{k}=0;
            end
            k=k+1;
            %coord1 is not always bigger than coord2, but it
            %doesn't matter
            if(min_plane_index>max_plane_index)
                step=-1;
            else
                step=1;
            end
            for pixels=min_plane_index+1:step:max_plane_index-1
                weight{k}=dp/(coord2-coord1);
                k=k+1;
            end
            weight{k}=(max_coord-max_plane)/(coord2-coord1);
            if(abs(weight{k})<tol_min)
                weight{k}=0;
            end
            k=k+1;
        else
            if((plane(index1)<max(coord1,coord2) || abs(plane(index1)-max(coord1,coord2))<tol_min)...
                    && (plane(index1)>min(coord1,coord2) ||abs(plane(index1)-min(coord1,coord2))<tol_min))
                max_plane=plane(index1);
            else
                max_plane=plane(index2);
            end
            assert((max_plane>min(coord1,coord2) || abs(max_plane-min(coord1,coord2))<tol_min)...
                && (max_plane<max(coord1,coord2) || abs(max_plane-max(coord1,coord2))<tol_min))
%             max_plane=max(plane(index1),plane(index2));
            weight{k}=(max_plane-coord1)/(coord2-coord1);
            if(abs(weight{k})<tol_min)
                weight{k}=0;
            end
            k=k+1;
            weight{k}=(coord2-max_plane)/(coord2-coord1);
            if(abs(weight{k})<tol_min)
                weight{k}=0;
            end
            k=k+1;
        end
    end
end
function TotalWeight=PixelWeightCalculator(coord1_1,coord1_2,plane1,dp1,coord2_1,coord2_2,plane2,dp2)
    weights1=WeightCalculator(coord1_1,coord1_2,plane1,dp1);
    weights2=WeightCalculator(coord2_1,coord2_2,plane2,dp2);
    TotalWeight=zeros(length(weights1),length(weights2));
    for i=1:length(weights2)
        for j=1:length(weights1)
            TotalWeight(j,i)=weights1{j}*weights2{i};
        end
    end
end