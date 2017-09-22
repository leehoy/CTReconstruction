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
    ki=((0:nu-1)-(nu-1)/2)*deltaS;
    p=((0:nv-1)-(nv-1)/2)*deltaS;
    [pp,kk]=meshgrid(p,ki);
    weight=R./(sqrt(R^2+kk.^2+pp.^2));
    ZeroPaddedLength=2^nextpow2(2*(nu-1));
    cutoff=params.cutoff;
    FilterType=params.FilterType;
    filter=FilterLine(ZeroPaddedLength+1,deltaS,FilterType,cutoff)*0.5;
    ReconSpacingX=params.ReconSpacingX; % fov/nx;
    ReconSpacingY=-1*params.ReconSpacingY;
    ReconSpacingZ=-1*params.ReconSpacingZ;
    recon_planeX=(-nx/2+(0:nx))*ReconSpacingX; % pixel boundaries of image
    recon_planeY=(-ny/2+(0:ny))*ReconSpacingY;
    recon_palneZ=(-nz/2+(0:nz))*ReconSpacingZ;
    recon_planeX=recon_planeX-ReconSpacingX/2;
    recon_planeY=recon_planeY-ReconSpacingY/2;
    recon_planeZ=recon_palneZ-ReconSpacingZ/2;
    x=(-(nx-1)/2:(nx-1)/2)*ReconSpacingX;
    y=(-(ny-1)/2:(ny-1)/2)*ReconSpacingY;
    z=(-(nz-1)/2:(nz-1)/2)*ReconSpacingZ;
    [Y,X]=meshgrid(y,x);
    [phi,r]=cart2pol(X,Y);
    recon=zeros(nx,ny,nz);
    theta=linspace(0,360,nViews+1);
    theta=theta*(pi/180);
    dtheta=(pi*2)/nViews;
    for i=1:nViews
        R1=proj(:,:,i);
        R2=weight.*R1;
        Q=zeros(size(R1));
        angle=theta(i);
        for k=1:size(R2,2)
            tmp=real(ifft(ifftshift(fftshift(fft(R2(:,k),ZeroPaddedLength)).*filter)));
            Q(:,k)=tmp(1:size(Q,1));%deltaS;
        end
%         InterpW=(R^2)./(
%         U=(R+r.*sin(angle-phi))./R;
%         R2=w.*R1;
        
        recon=recon+backproj(Q,recon_planeX,recon_planeY,recon_planeZ,angle,R,R+D,nu,nv,du,dv)*dtheta;
%         imshow(recon(:,:,128),[]);
%         pause(0.04);
    end
    imshow(recon(:,:,128),[]);
end
function bp=backproj(proj,recon_planeX,recon_planeY,recon_planeZ,angle,SAD,SDD,nu,nv,du,dv)
    tol_min=1e-6;
    nx=length(recon_planeX)-1;
    ny=length(recon_planeY)-1;
    nz=length(recon_planeZ)-1;
    bp=zeros(nx,ny,nz);
    dx=recon_planeX(2)-recon_planeX(1);
    dy=recon_planeY(2)-recon_planeY(1);
    dz=recon_planeZ(2)-recon_planeZ(1);
    recon_PixelsX=recon_planeX(1:end-1)+dx/2; %x center of the pixels
    recon_PixelsY=recon_planeY(1:end-1)+dy/2; %y center of the pixels
    recon_PixelsZ=recon_planeZ(1:end-1)+dz/2; %y center of the pixels
%     recon_PixelsX=recon_planeX(1:end-1); %x center of the pixels
%     recon_PixelsY=recon_planeY(1:end-1); %y center of the pixels
    for h=1:nz
        for i=1:nx
            for j=1:ny
%                 Z-direction rotation assumed
                xc=(recon_PixelsX(i))*cos(angle)+(recon_PixelsY(j))*sin(angle);
                yc=-(recon_PixelsX(i))*sin(angle)+(recon_PixelsY(j))*cos(angle);
%                 zc=recon_PixelsZ(h);
                x1=(recon_PixelsX(i)+dx/2)*cos(angle)+(recon_PixelsY(j)+dy/2)*sin(angle);
                y1=-(recon_PixelsX(i)+dx/2)*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
%                 z1=recon_PixelsZ(h)-dz/2;
                x2=(recon_PixelsX(i)-dx/2)*cos(angle)+(recon_PixelsY(j)-dy/2)*sin(angle);
                y2=-(recon_PixelsX(i)-dx/2)*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
%                 z2=recon_PixelsZ(h)-dz/2;
                x3=(recon_PixelsX(i)+dx/2)*cos(angle)+(recon_PixelsY(j)-dy/2)*sin(angle);
                y3=-(recon_PixelsX(i)+dx/2)*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
%                 z3=recon_PixelsZ(h)-dz/2;
                x4=(recon_PixelsX(i)-dx/2)*cos(angle)+(recon_PixelsY(j)+dy/2)*sin(angle);
                y4=-(recon_PixelsX(i)-dx/2)*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
%                 z4=recon_PixelsZ(h)-dz/2;
                
                x5=(recon_PixelsX(i)+dx/2)*cos(angle)+(recon_PixelsY(j)+dy/2)*sin(angle);
                y5=-(recon_PixelsX(i)+dx/2)*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
%                 z5=recon_PixelsZ(h)+dz/2;
                x6=(recon_PixelsX(i)-dx/2)*cos(angle)+(recon_PixelsY(j)-dy/2)*sin(angle);
                y6=-(recon_PixelsX(i)-dx/2)*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
%                 z6=recon_PixelsZ(h)+dz/2;
                x7=(recon_PixelsX(i)+dx/2)*cos(angle)+(recon_PixelsY(j)-dy/2)*sin(angle);
                y7=-(recon_PixelsX(i)+dx/2)*sin(angle)+(recon_PixelsY(j)-dy/2)*cos(angle);
%                 z7=recon_PixelsZ(h)+dz/2;
                x8=(recon_PixelsX(i)-dx/2)*cos(angle)+(recon_PixelsY(j)+dy/2)*sin(angle);
                y8=-(recon_PixelsX(i)-dx/2)*sin(angle)+(recon_PixelsY(j)+dy/2)*cos(angle);
%                 z8=recon_PixelsZ(h)+dz/2;

                x_set=[x1,x2,x3,x4];
                y_set=[y1,y2,y3,y4];
                xu1=min(x_set);
                yu1=y_set(x_set==xu1);
                if(length(yu1)>1)
                    yu1=yc;
                end
                xu2=max(x_set);
                yu2=y_set(x_set==xu2);
                if(length(yu2)>1)
                    yu2=yc;
                end
                zv1=recon_PixelsZ(h)+sqrt((dy/2)^2+(dz/2)^2);
                zv2=recon_PixelsZ(h)-sqrt((dy/2)^2+(dz/2)^2);
                slopes_v=[zv1/(-y1+SAD),zv1/(-y2+SAD),zv1/(-y3+SAD),zv1/(-y4+SAD),...
                    zv2/(-y1+SAD),zv2/(-y2+SAD),zv2/(-y3+SAD),zv2/(-y4+SAD)];
%                 zv1=min(z_set);
%                 yv1=y_set(z_set==zv1);
%                 if(length(yv1)>1)
%                     yv1=yc;
%                 end
%                 zv2=max(z_set);
%                 yv2=y_set(z_set==zv2);
%                 if(length(yv2)>1)
%                     yv2=yc;
%                 end
                u_l=((xu1)/(-yu1+SAD))*SDD/du+nu/2;
                u_r=((xu2)/(-yu2+SAD))*SDD/du+nu/2;
%                 u_l=((xc-dx/2)/(-yc+SAD))*SDD/du+nu/2;
%                 u_r=((xc+dx/2)/(-yc+SAD))*SDD/du+nu/2;
                u_min=min(u_l,u_r); 
                u_max=max(u_l,u_r);                
                v_l=(max(slopes_v))*SDD/dv+nv/2;
                v_r=(min(slopes_v))*SDD/dv+nv/2;
                v_min=min(v_l,v_r);
                v_max=max(v_l,v_r);
%                 for k=floor(u_min):floor(u_max)
%                     if(k<1 || k>nu)
%                         continue;
%                     end
%                     assert(u_min~=u_max)
%                     if(ceil(u_min)==floor(u_max))
%                         weight2=1.0;
%                     elseif(k==floor(u_min))
%                        weight2=(ceil(u_min)-u_min)/(u_max-u_min);
%                     elseif(k==floor(u_max))
%                        weight2=(u_max-floor(u_max))/(u_max-u_min);
%                     else
%                        weight2=du/(u_max-u_min);
%                     end
%                     if(abs(weight2)<tol_min)
%                         weight2=0;
%                     end
%                     bp(i,j,h)=bp(i,j,h)+proj(k,192)*1*weight2;
%                 end
                for l=floor(v_min):floor(v_max)
                    if(l<1 || l>nv)
                        continue;
                    end
                    assert(v_min~=v_max);
                    if(ceil(v_min)==floor(v_max))
                        weight1=1.0;
                    elseif(l==floor(v_min))
                        weight1=(ceil(v_min)-v_min)/(v_max-v_min);
                    elseif(l==floor(v_max))
                        weight1=(v_max-floor(v_max))/(v_max-v_min);
                    else
                        weight1=dv/(v_max-v_min);
                    end
                    if(abs(weight1)<tol_min)
                        weight1=0;
                    end
                    for k=floor(u_min):floor(u_max)
                        if(k<1 || k>nu)
                            continue;
                        end
                        assert(u_min~=u_max)
                        if(ceil(u_min)==floor(u_max))
                            weight2=1.0;
                        elseif(k==floor(u_min))
                           weight2=(ceil(u_min)-u_min)/(u_max-u_min);
                        elseif(k==floor(u_max))
                           weight2=(u_max-floor(u_max))/(u_max-u_min);
                        else
                           weight2=du/(u_max-u_min);
                        end
                        if(abs(weight2)<tol_min)
                            weight2=0;
                        end
                        bp(i,j,h)=bp(i,j,h)+proj(k,l)*weight1*weight2;
                    end
                end
            end
        end

    end
end