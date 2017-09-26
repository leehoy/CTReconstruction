function [ recon ] = DistanceDrivenBackprojection3D( proj,params )
%UNTITLED6 이 함수의 요약 설명 위치
%   자세한 설명 위치
    nu=params.nu;
    nv=params.nv;
    nViews=params.nview;
    R=params.SourceToAxis;
    D=params.SourceToDetector-R;
    du=params.du;
    dv=-1*params.dv;
    deltaS=du*R/(R+D);
    nx=params.nx;
    ny=params.ny;
    nz=params.nz;
    ki=((1:nu)-(nu-1)/2)*du;
    p=((1:nv)-(nv-1)/2)*dv;
%     ki=((0:nu)-(nu-1)/2)*du;%*R/(R+D);
%     p=((0:nv)-(nv-1)/2)*dv;%*R/(R+D);
%     ki=((0:nu)-nu/2)*du-du/2;
%     p=((0:nv)-nv/2)*dv-dv/2;
%     [pp,kk]=meshgrid(p(1:end-1).*R/(R+D),ki(1:end-1)*R/(R+D));
%     weight=R./(sqrt(R^2+kk.^2+pp.^2));
    [pp,kk]=meshgrid(p*R/(R+D),ki*R/(R+D));
    weight=R./(sqrt(R^2+kk.^2+pp.^2));
    ZeroPaddedLength=2^nextpow2(2*(nu-1));
    cutoff=params.cutoff;
    FilterType=params.FilterType;
    filter=FilterLine(ZeroPaddedLength+1,deltaS,FilterType,cutoff);
    ReconSpacingX=params.ReconSpacingX; % fov/nx;
    ReconSpacingY=-1*params.ReconSpacingY;
    ReconSpacingZ=-1*params.ReconSpacingZ;
    recon_planeX=(-nx/2+(0:nx))*ReconSpacingX; % pixel boundaries of image
    recon_planeY=(-ny/2+(0:ny))*ReconSpacingY;
    recon_planeZ=(-nz/2+(0:nz))*ReconSpacingZ;
    recon_planeX=recon_planeX-ReconSpacingX/2;
    recon_planeY=recon_planeY-ReconSpacingY/2;
    recon_planeZ=recon_planeZ-ReconSpacingZ/2;
    recon_pixelsX=recon_planeX(1:end-1)+ReconSpacingX/2;
    recon_pixelsY=recon_planeY(1:end-1)+ReconSpacingY/2;
    recon_pixelsZ=recon_planeZ(1:end-1)+ReconSpacingZ/2;
%     x=(-(nx-1)/2:(nx-1)/2)*ReconSpacingX;
%     y=(-(ny-1)/2:(ny-1)/2)*ReconSpacingY;
%     z=(-(nz-1)/2:(nz-1)/2)*ReconSpacingZ;
    [X,Y]=meshgrid(recon_pixelsX,recon_pixelsY);
    recon=zeros(nx,ny,nz);
    theta=linspace(0,360,nViews+1);
    theta=theta*(pi/180);
    dtheta=(pi*2)/nViews;
    Source=[-R*sin(theta(1)),R*cos(theta(1)),0];
    Detector=[D*sin(theta(1)),-D*cos(theta(1)),0];
    u_plane=[ki-du/2,ki(end)+du/2];
    v_plane=[p-dv/2,p(end)+dv/2];
    for i=1:nViews
        R1=proj(:,:,i);
        R2=weight.*R1;
        Q=zeros(size(R1));
        angle=theta(i);
        t=X.*cos(angle)+Y.*sin(angle);
        s=-X.*sin(angle)+Y.*cos(angle);
        for k=1:nv
            tmp=real(ifft(ifftshift(fftshift(fft(R2(:,k),ZeroPaddedLength)).*filter)));
            Q(:,k)=tmp(1:nu)*deltaS;
        end
        InterpW=(R^2)./(R-t).^2;
%         imshow(InterpW,[]);
        InterpW=repmat(InterpW,[1,1,nz]);
        recon=recon+backproj(Q,recon_pixelsX,recon_pixelsY,recon_pixelsZ,angle,R,R+D,Source,Detector,u_plane,v_plane).*InterpW*dtheta;
        imshow(recon(:,:,128),[]);
        
        fprintf('%04d\n',i);
    end
%     imshow(recon(:,:,82),[]);
end
function bp=backproj(proj,recon_pixelsX,recon_pixelsY,recon_pixelsZ,angle,SAD,SDD,Source,Detector,u_plane,v_plane)
    tol_min=1e-6;
    nx=length(recon_pixelsX);
    ny=length(recon_pixelsY);
    nz=length(recon_pixelsZ);
    bp=zeros(nx,ny,nz);
    nu=length(u_plane)-1;
    nv=length(v_plane)-1;
    du=u_plane(2)-u_plane(1);
    dv=v_plane(2)-v_plane(1);
    dx=recon_pixelsX(2)-recon_pixelsX(1);
    dy=recon_pixelsY(2)-recon_pixelsY(1);
    dz=recon_pixelsZ(2)-recon_pixelsZ(1);
%     recon_PixelsX=recon_planeX(1:end-1)+dx/2; %x center of the pixels
%     recon_PixelsY=recon_planeY(1:end-1)+dy/2; %y center of the pixels
%     recon_PixelsZ=recon_planeZ(1:end-1)+dz/2; %y center of the pixels
%     [reconX,reconY]=meshgrid(recon_PixelsX,recon_PixelsY);
%     reconX_c1=(reconX+dx/2)*cos(angle)+(reconY+dy/2)*sin(angle);
%     reconY_c1=-(reconX+dx/2)*sin(angle)+(reconY+dy/2)*cos(angle);
%     reconX_c2=(reconX-dx/2)*cos(angle)+(reconY-dy/2)*sin(angle);
%     reconY_c2=-(reconX-dx/2)*sin(angle)+(reconY-dy/2)*cos(angle);
%     reconX_c3=(reconX+dx/2)*cos(angle)+(reconY-dy/2)*sin(angle);
%     reconY_c3=-(reconX+dx/2)*sin(angle)+(reconY-dy/2)*cos(angle);
%     reconX_c4=(reconX-dx/2)*cos(angle)+(reconY+dy/2)*sin(angle);
%     reconY_c4=-(reconX-dx/2)*sin(angle)+(reconY+dy/2)*cos(angle);
%     slopesU_c1=(Source(1)-reconX_c1)./(Source(2)-reconY_c1);
%     slopesU_c2=(Source(1)-reconX_c2)./(Source(2)-reconY_c2);
%     slopesU_c3=(Source(1)-reconX_c3)./(Source(2)-reconY_c3);
%     slopesU_c4=(Source(1)-reconX_c4)./(Source(2)-reconY_c4);
%     slopesV_c1=(reconZ+sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c1+SAD);
%     slopesV_c2=(reconZ+sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c2+SAD);
%     slopesV_c3=(reconZ+sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c3+SAD);
%     slopesV_c4=(reconZ+sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c4+SAD);
%     slopesV_c5=(reconZ-sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c1+SAD);
%     slopesV_c6=(reconZ-sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c2+SAD);
%     slopesV_c7=(reconZ-sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c3+SAD);
%     slopesV_c8=(reconZ-sqrt((dy/2)^2+(dz/2)^2))./(-reconY_c4+SAD);
%     recon_PixelsX=recon_planeX(1:end-1); %x center of the pixels
%     recon_PixelsY=recon_planeY(1:end-1); %y center of the pixels
    for h=128:128
%         slopesV_c1=(Source(3)-(recon_PixelsZ(h)+sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c1);
%         slopesV_c2=(Source(3)-(recon_PixelsZ(h)+sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c2);
%         slopesV_c3=(Source(3)-(recon_PixelsZ(h)+sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c3);
%         slopesV_c4=(Source(3)-(recon_PixelsZ(h)+sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c4);
%         slopesV_c5=(Source(3)-(recon_PixelsZ(h)-sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c1);
%         slopesV_c6=(Source(3)-(recon_PixelsZ(h)-sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c2);
%         slopesV_c7=(Source(3)-(recon_PixelsZ(h)-sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c3);
%         slopesV_c8=(Source(3)-(recon_PixelsZ(h)-sqrt((dy/2)^2+(dz/2)^2)))./(Source(2)-reconY_c4);
        for i=1:nx
            for j=1:ny
                xc=recon_pixelsX(i)*cos(angle)+recon_pixelsY(j)*sin(angle);
                yc=-recon_pixelsX(i)*sin(angle)+recon_pixelsY(j)*cos(angle);
                zc=recon_pixelsZ(h);
%                 Z-direction rotation assumed
%                 x1=(recon_pixelsX(i)+dx/2)*cos(angle)+(recon_pixelsY(j)+dy/2)*sin(angle);
%                 y1=-(recon_pixelsX(i)+dx/2)*sin(angle)+(recon_pixelsY(j)+dy/2)*cos(angle);
%                 x2=(recon_pixelsX(i)-dx/2)*cos(angle)+(recon_pixelsY(j)-dy/2)*sin(angle);
%                 y2=-(recon_pixelsX(i)-dx/2)*sin(angle)+(recon_pixelsY(j)-dy/2)*cos(angle);
%                 x3=(recon_pixelsX(i)+dx/2)*cos(angle)+(recon_pixelsY(j)-dy/2)*sin(angle);
%                 y3=-(recon_pixelsX(i)+dx/2)*sin(angle)+(recon_pixelsY(j)-dy/2)*cos(angle);
%                 x4=(recon_pixelsX(i)-dx/2)*cos(angle)+(recon_pixelsY(j)+dy/2)*sin(angle);
%                 y4=-(recon_pixelsX(i)-dx/2)*sin(angle)+(recon_pixelsY(j)+dy/2)*cos(angle);
%                 slopesU=[(Source(1)-x1)/(Source(2)-y1),(Source(1)-x2)/(Source(2)-y2),...
%                     (Source(1)-x3)/(Source(2)-y3),(Source(1)-x4)/(Source(2)-y4)];
                slopesU=[(Source(1)-(xc-dx/2))/(Source(2)-yc),(Source(1)-(xc+dx/2))/(Source(2)-yc)];
                coord_u1=((min(slopesU)*Detector(2))+(Source(1)-min(slopesU)*Source(2)));
                coord_u2=((max(slopesU)*Detector(2))+(Source(1)-max(slopesU)*Source(2)));
                u_l=floor((coord_u1-u_plane(1))/du+1);
                u_r=floor((coord_u2-u_plane(1))/du+1);
                s_index_u=min(u_l,u_r);
                e_index_u=max(u_l,u_r);
                z1=recon_pixelsZ(h)+dz/2;
                z2=recon_pixelsZ(h)-dz/2;
%                 y1=-(recon_PixelsX(i))*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
%                 y2=-(recon_PixelsX(i))*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
%                 slopesV=[z1/(-y1+SAD),z1/(-y2+SAD),z2/(-y1+SAD),z2/(-y2+SAD)];
%                 slopesV=[(Source(3)-z1)/(Source(2)-y1),(Source(3)-z1)/(Source(2)-y2),...
%                     (Source(3)-z1)/(Source(2)-y3),(Source(3)-z1)/(Source(2)-y4),...
%                     (Source(3)-z2)/(Source(2)-y1),(Source(3)-z2)/(Source(2)-y2),...
%                     (Source(3)-z2)/(Source(2)-y3),(Source(3)-z2)/(Source(2)-y4)];
                slopesV=[(Source(3)-z1)/(Source(2)-yc),(Source(3)-z2)/(Source(2)-yc)];
                coord_v1=((max(slopesV)*Detector(2))+(Source(3)-max(slopesV)*Source(2)));
                coord_v2=((min(slopesV)*Detector(2))+(Source(3)-min(slopesV)*Source(2)));
                v_l=floor((coord_v1-v_plane(1))/dv+1);
                v_r=floor((coord_v2-v_plane(1))/dv+1);
                s_index_v=min(v_l,v_r);
                e_index_v=max(v_l,v_r);

                for l=s_index_v:e_index_v
                    if(l<1 || l>nv)
                        continue;
                    end
                    assert(coord_v2~=coord_v1);
                    if(s_index_v==e_index_v)
                        weight1=1.0;
                    elseif(l==s_index_v)
                        weight1=(max(coord_v2,coord_v1)-v_plane(l+1))/abs(coord_v1-coord_v2);
                    elseif(l==e_index_v)
                        weight1=(v_plane(l)-min(coord_v2,coord_v1))/abs(coord_v1-coord_v2);
                    else
                        weight1=abs(dv)/abs(coord_v1-coord_v2);
                    end
                    if(abs(weight1)<tol_min)
                        weight1=0;
                    end
                    for k=s_index_u:e_index_u
                        if(k<1 || k>nu)
                            continue;
                        end
                        assert(coord_u1~=coord_u2)
                        if(s_index_u==e_index_u)
                            weight2=1.0;
                        elseif(k==s_index_u)
                           weight2=(u_plane(k+1)-min(coord_u1,coord_u2))/abs(coord_u2-coord_u1);
                        elseif(k==e_index_u)
                           weight2=(max(coord_u1,coord_u2)-u_plane(k))/abs(coord_u2-coord_u1);
                        else
                           weight2=abs(du)/abs(coord_u2-coord_u1);
                        end
                        if(abs(weight2)<tol_min)
                            weight2=0;
                        end
                        assert( weight1>=0 && weight2>=0 && weight1<=1 && weight2<=1);
                        bp(j,i,h)=bp(j,i,h)+proj(k,l)*weight1*weight2;%*(SAD^2)/(SAD^2-yc);
                    end
                end
            end
        end
    end
end