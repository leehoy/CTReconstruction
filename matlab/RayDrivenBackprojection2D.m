function [ recon ] = RayDrivenBackprojection2D( proj,params )
%UNTITLED6 이 함수의 요약 설명 위치
%   자세한 설명 위치
    N=size(proj,1);
    M=size(proj,2);
%     DetectorWidth=N;
    R=params.SourceToAxis;
    D=params.SourceToDetector-R;
    r_spacing=params.DetectorPixelSpacing;
    deltaS=r_spacing*R/(R+D);
    SourceCenter=[0,R];
    DetectorCenter=[0,-D];
%     Source and detector position and direction must be matched to
%     projection code
    nx=params.nx;
    ny=params.ny;
%     DetectorSize=r_spacing*DetectorWidth;
    gamma=((0:N-1)-(N-1)/2)*deltaS;
    ZeroPaddedLength=2^nextpow2(2*(N-1));
    cutoff=params.cutoff;
    FilterType=params.FilterType;
    filter=FilterLine(ZeroPaddedLength+1,deltaS,FilterType,cutoff)*0.5;
%     fov=2*R*sin(atan((DetectorSize/2)/(R+D)));
    ReconSpacingX=params.ReconSpacingX; % fov/nx;
    ReconSpacingY=params.ReconSpacingY;
    recon_planeX=(-nx/2+(0:nx))*ReconSpacingX; % pixel boundaries of image
    recon_planeY=(-ny/2+(0:ny))*ReconSpacingY;
    recon_planeX=recon_planeX-ReconSpacingX/2;
    recon_planeY=recon_planeY-ReconSpacingY/2;
    x=(-(nx-1)/2:(nx-1)/2)*ReconSpacingX;
%     y=(ny-1)/2:-1:-(ny-1)/2;
    y=((ny-1)/2:-1:-(ny-1)/2)*ReconSpacingY;
%     [X,Y]=meshgrid(x,y);
    [Y,X]=meshgrid(y,x);
    [phi,r]=cart2pol(X,Y);
%     Weight U need to have same direction with back projection angle?
    recon=zeros(nx,ny);
    theta=linspace(0,360,M+1);
    theta=theta*(pi/180);
    dtheta=(pi*2)/M;
    for i=1:M
        R1=proj(:,i);
        w=((R)./sqrt((R)^2+gamma'.^2));
        angle=theta(i);
        U=(R+r.*sin(angle-phi))./R;
        R2=w.*R1;
        Q=real(ifft(ifftshift(fftshift(fft(R2,ZeroPaddedLength)).*filter)));
        Q=Q(1:length(R2))*deltaS;
        recon=recon+backproj(Q,recon_planeX,recon_planeY,angle,R,R+D,N,r_spacing,SourceCenter,DetectorCenter)./(U.^2)*dtheta;
%         imshow(U,[]);
%         pause(0.01);
    end
    close all;
    imshow(recon,[]);
end
function bp=backproj(proj,recon_planeX,recon_planeY,angle,SAD,SDD,nd,DetectorPixelSpacing,Source,Detector)
    nx=length(recon_planeX)-1;
    ny=length(recon_planeY)-1;
    SourceX=Source(1);
    SourceY=Source(2);
    DetectorX=Detector(1);
    DetectorY=Detector(2);
    bp=zeros(nx,ny);
    dx=recon_planeX(2)-recon_planeX(1);
    dy=recon_planeY(2)-recon_planeY(1);
    recon_PixelsX=recon_planeX(1:end-1)+dx/2; %x center of the pixels
    recon_PixelsY=recon_planeY(1:end-1)+dy/2; %y center of the pixels
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
            
            x_set=[x1,x2,x3,x4];
            y_set=[y1,y2,y3,y4];
            xl=min(x_set);
            yl=y_set(x_set==xl);
            if(length(yl)>1)
                yl=yc;
            end
            xr=max(x_set);
            yr=y_set(x_set==xr);
            if(length(yr)>1)
                yr=yc;
            end
            n_l=((xl)/(-yl+SAD))*SDD/DetectorPixelSpacing+nd/2;
            n_r=((xr)/(-yr+SAD))*SDD/DetectorPixelSpacing+nd/2;
            n_min=min(n_l,n_r);
            n_max=max(n_l,n_r);
            for k=floor(n_min):floor(n_max)
                if(k<1 || k>nd)
                    continue;
                end
                assert(n_min~=n_max)
%                 y1=DetectorY+(k-nd/2)*DetectorPixelSpacing;
                x1=DetectorX+(k-nd/2)*DetectorPixelSpacing;
%                 slope=(y1-SourceY)/(DetectorX-SourceX);
                slope=(x1-SourceX)/(DetectorY-SourceY);
                intercept=-slope*SourceY+SourceX;
                b=sqrt(slope^2/(1+slope^2));
                if(slope~=0)
                    a=-1*sign(slope)*sqrt(1-b^2);
                else
                    a=sqrt(1-b^2);
                end
                c=-a*intercept;
                a2=abs(a);
                b2=abs(b);
                d=abs(a*xc+b*yc+c)/sqrt(a^2+b^2);
                d1=abs(a2-b2)/2;
                d2=(a2+b2)/2;
                if(d<d1 && a2<b2)
                    l=1/b2;
                elseif(d<d1&& a2>=b2)
                    l=1/a2;
                elseif(d>=d1&& d<d2)
                    l=(d2-d)/(a2*b2);
                elseif(d>=d2)
                    l=0;
                end
                if(ceil(n_min)==floor(n_max))
                    bp(j,i)=proj(k)*1*l;
                    continue;
                end
                if(k==floor(n_min))
                    bp(j,i)=proj(k)*1*l;
                elseif(k>floor(n_min) && k<floor(n_max))
                    bp(j,i)=proj(k)*1*l;
                elseif(k==floor(n_max))
                    bp(j,i)=proj(k)*1*l;
                else
                    fprintf('????\n');
                end
            end
        end
    end
end