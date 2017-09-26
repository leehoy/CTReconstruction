function [ recon ] = DistanceDrivenBackprojection2D( proj,params )
%UNTITLED6 이 함수의 요약 설명 위치
%   자세한 설명 위치
    N=size(proj,1);
    M=size(proj,2);
    R=params.SourceToAxis;
    D=params.SourceToDetector-R;
    r_spacing=params.DetectorPixelSpacing;
    deltaS=r_spacing*R/(R+D);
    SourceCenter=[0,R];
    DetectorCenter=[0,-D];
    nx=params.nx;
    ny=params.ny;
    gamma=((0:N-1)-(N-1)/2)*deltaS;
    detector_pixels=((0:N)-N/2)*r_spacing+r_spacing/2;
    ZeroPaddedLength=2^nextpow2(2*(N-1));
    cutoff=params.cutoff;
    FilterType=params.FilterType;
    filter=FilterLine(ZeroPaddedLength+1,deltaS,FilterType,cutoff);
    ReconSpacingX=params.ReconSpacingX; % fov/nx;
    ReconSpacingY=params.ReconSpacingY;
    recon_planeX=(-nx/2+(0:nx))*ReconSpacingX; % pixel boundaries of image
    recon_planeY=(-ny/2+(0:ny))*ReconSpacingY;
    recon_planeX=recon_planeX-ReconSpacingX/2;
    recon_planeY=recon_planeY-ReconSpacingY/2;
    x=(-(nx-1)/2:(nx-1)/2)*ReconSpacingX;
    y=((ny-1)/2:-1:-(ny-1)/2)*ReconSpacingY;
    [Y,X]=meshgrid(y,x);
    [phi,r]=cart2pol(X,Y);
    recon=zeros(nx,ny);
    theta=linspace(0,360,M+1);
    theta=theta*(pi/180);
    dtheta=(pi*2)/M;
    weight_map=zeros(nx,ny,M);
    for i=1:M
        R1=proj(:,i);
        w=((R)./sqrt((R)^2+gamma'.^2));
        angle=theta(i);
        U=(R+r.*sin(angle-phi))./R;
        R2=w.*R1;
        Q=real(ifft(ifftshift(fftshift(fft(R2,ZeroPaddedLength)).*filter)));
        Q=Q(1:length(R2))*deltaS;
        [bp,weight_map(:,:,i)]=backproj(Q,recon_planeX,recon_planeY,angle,SourceCenter,DetectorCenter,detector_pixels);
        recon=recon+bp./(U.^2)*dtheta;
        imshow(recon,[]);
    end
end
function [bp,weight_map]=backproj(proj,recon_planeX,recon_planeY,angle,Source,Detector,d)
    nx=length(recon_planeX)-1;
    ny=length(recon_planeY)-1;
    bp=zeros(nx,ny);
    dx=recon_planeX(2)-recon_planeX(1);
    dy=recon_planeY(2)-recon_planeY(1);
    dd=d(2)-d(1);
    nd=length(d)-1;
    recon_PixelsX=recon_planeX(1:end-1)+dx/2; %x center of the pixels
    recon_PixelsY=recon_planeY(1:end-1)+dy/2; %y center of the pixels
    weight_map=zeros(size(bp));
%     recon_PixelsX=recon_planeX(1:end-1); %x center of the pixels
%     recon_PixelsY=recon_planeY(1:end-1); %y center of the pixels
    for i=1:nx
        for j=1:ny
            xc=(recon_PixelsX(i))*cos(angle)+(recon_PixelsY(j))*sin(angle);
            yc=-(recon_PixelsX(i))*sin(angle)+(recon_PixelsY(j))*cos(angle);
            
            x1=(recon_PixelsX(i)+dx/2)*cos(angle)+(recon_PixelsY(j)+dy/2)*sin(angle);
            y1=-(recon_PixelsX(i)+dx/2)*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
            x2=(recon_PixelsX(i)-dx/2)*cos(angle)+(recon_PixelsY(j)-dy/2)*sin(angle);
            y2=-(recon_PixelsX(i)-dx/2)*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
            x3=(recon_PixelsX(i)+dx/2)*cos(angle)+(recon_PixelsY(j)-dy/2)*sin(angle);
            y3=-(recon_PixelsX(i)+dx/2)*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
            x4=(recon_PixelsX(i)-dx/2)*cos(angle)+(recon_PixelsY(j)+dy/2)*sin(angle);
            y4=-(recon_PixelsX(i)-dx/2)*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
            
%             x_set=[x1,x2,x3,x4];
%             y_set=[y1,y2,y3,y4];
%             xl=min(x_set);
%             yl=y_set(x_set==xl);
%             if(length(yl)>1)
%                 yl=yc;
%             end
%             xr=max(x_set);
%             yr=y_set(x_set==xr);
%             if(length(yr)>1)
%                 yr=yc;
%             end
% %             n_l=((xl)/(-yl+SAD))*SDD/DetectorPixelSpacing+nd/2;
% %             n_r=((xr)/(-yr+SAD))*SDD/DetectorPixelSpacing+nd/2;
%             n_l=(xc-dx/2)/(-yc+SAD)*SDD/DetectorPixelSpacing+nd/2;
%             n_r=(xc+dx/2)/(-yc+SAD)*SDD/DetectorPixelSpacing+nd/2;
            slopes=[(Source(1)-x1)/(Source(2)-y1),(Source(1)-x2)/(Source(2)-y2),...
                (Source(1)-x3)/(Source(2)-y3),(Source(1)-x4)/(Source(2)-y4)];
            coord_l=(min(slopes)*Detector(2))+(Source(1)-min(slopes)*Source(2));
            coord_r=(max(slopes)*Detector(2))+(Source(1)-max(slopes)*Source(2));
            n_l=floor((((min(slopes)*Detector(2))+(Source(1)-min(slopes)*Source(2)))-d(1))/dd+1);
            n_r=floor((((max(slopes)*Detector(2))+(Source(1)-max(slopes)*Source(2)))-d(1))/dd+1);
%             n_min=min(n_l,n_r);
%             n_max=max(n_l,n_r);
            s_index=min(n_l,n_r);
            e_index=max(n_l,n_r);
            for k=s_index:e_index
                if(k<1 || k>nd)
                    continue;
                end
                assert(coord_l~=coord_r)
                if(s_index==e_index)
                    weight=1;
%                     bp(j,i)=bp(j,i)+proj(k)*1;
                elseif(k==s_index)
                    weight=(d(k+1)-min(coord_l,coord_r))/abs(coord_l-coord_r);
%                     bp(j,i)=bp(j,i)+proj(k)*(ceil(n_min)-n_min)/abs(n_max-n_min);
                elseif(k==e_index)
                    weight=(max(coord_l,coord_r)-d(k))/abs(coord_l-coord_r);
%                     bp(j,i)=bp(j,i)+proj(k)*(n_max-floor(n_max))/abs(n_max-n_min);
                else
                    weight=abs(dd)/abs(coord_l-coord_r);
%                     bp(j,i)=bp(j,i)+proj(k)*dd/abs(n_max-n_min);
                end
                assert(weight>0 && weight<=1);
                bp(j,i)=bp(j,i)+proj(k)*weight;
                weight_map(j,i)=weight;
            end
        end
    end
end