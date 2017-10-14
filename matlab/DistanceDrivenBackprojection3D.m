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
        for k=1:nv
            tmp=real(ifft(ifftshift(fftshift(fft(R2(:,k),ZeroPaddedLength)).*filter)));
            Q(:,k)=tmp(1:nu)*deltaS;
        end
        recon=recon+backproj(Q,recon_pixelsX,recon_pixelsY,recon_pixelsZ,angle,R,R+D,Source,Detector,u_plane,v_plane)*dtheta;
        
        fprintf('%04d\n',i);
    end
end
function bp=backproj(proj,recon_pixelsX,recon_pixelsY,recon_pixelsZ,angle,SAD,Source,Detector,u_plane,v_plane)
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
    for h=1:nx
        for i=1:nx
            for j=1:ny
                yc=-recon_pixelsX(i)*sin(angle)+recon_pixelsY(j)*cos(angle);
%                 Z-direction rotation assumed
                x1=(recon_pixelsX(i)+dx/2)*cos(angle)+(recon_pixelsY(j)+dy/2)*sin(angle);
                y1=-(recon_pixelsX(i)+dx/2)*sin(angle)+(recon_pixelsY(j)+dy/2)*cos(angle);
                x2=(recon_pixelsX(i)-dx/2)*cos(angle)+(recon_pixelsY(j)-dy/2)*sin(angle);
                y2=-(recon_pixelsX(i)-dx/2)*sin(angle)+(recon_pixelsY(j)-dy/2)*cos(angle);
                x3=(recon_pixelsX(i)+dx/2)*cos(angle)+(recon_pixelsY(j)-dy/2)*sin(angle);
                y3=-(recon_pixelsX(i)+dx/2)*sin(angle)+(recon_pixelsY(j)-dy/2)*cos(angle);
                x4=(recon_pixelsX(i)-dx/2)*cos(angle)+(recon_pixelsY(j)+dy/2)*sin(angle);
                y4=-(recon_pixelsX(i)-dx/2)*sin(angle)+(recon_pixelsY(j)+dy/2)*cos(angle);
                slopesU=[(Source(1)-x1)/(Source(2)-y1),(Source(1)-x2)/(Source(2)-y2),...
                    (Source(1)-x3)/(Source(2)-y3),(Source(1)-x4)/(Source(2)-y4)];
                coord_u1=((min(slopesU)*Detector(2))+(Source(1)-min(slopesU)*Source(2)));
                coord_u2=((max(slopesU)*Detector(2))+(Source(1)-max(slopesU)*Source(2)));
                u_l=floor((coord_u1-u_plane(1))/du+1);
                u_r=floor((coord_u2-u_plane(1))/du+1);
                s_index_u=min(u_l,u_r);
                e_index_u=max(u_l,u_r);
                z1=recon_pixelsZ(h)+dz/2;
                z2=recon_pixelsZ(h)-dz/2;
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
                        bp(i,j,h)=bp(i,j,h)+proj(k,l)*weight1*weight2*(SAD^2)/(SAD-yc)^2;
                    end
                end
            end
        end
    end
end