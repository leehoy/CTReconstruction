close all;
N=size(F1,1);
M=size(F1,2);
r_spacing=0.05;
nx=512;
ny=nx;
r_max=floor(N/2)*r_spacing;
gamma=-r_max:r_spacing:r_max;
gamma=gamma*pi/180;
cutoff=0.5;
smooth=0.6;
if(mod(N,2)==0)
    h=FilterArc(N,r_spacing*pi/180,cutoff,smooth);
    h=h(1:N-1);
else
    h=FilterArc(N,r_spacing*pi/180,cutoff,smooth);
end

% n=-floor(N/2):floor(N/2);
% h=FanFilter((1:N)-(N-1)/2,r_spacing*pi/180);

filter=abs(fftshift(fft(h)));
D=500;
x=1:nx;
y=1:ny;
mid=nx/2;
[X,Y]=meshgrid(x,y);
xpr=X-mid;
ypr=Y-mid;
recon=zeros(nx,ny);
[phi,r]=cart2pol(xpr,ypr);
theta=2*pi:-2*pi/M:0;
dtheta=abs(theta(1)-theta(2));
for i=1:M
    R=F1(:,i);
%     R2=D*R.*cosd(5)';
    R2=(D*r_spacing*pi/180)*R.*cos(gamma)';
%     R2=(R./(D*r_spacing)).*cosd(gamma)';
    Q=real(ifft(ifftshift(fftshift(fft(R2)).*filter)));
    angle=theta(i);
    gamma2=atan((r.*cos(angle-phi))./((D)+r.*sin(angle-phi)));
    ii=find((gamma2>min(gamma(:)))&(gamma2<max(gamma(:))));
    gamma2=gamma2(ii);
    L=sqrt((D+r(ii).*sin(angle-phi(ii))).^2+(r(ii).*cos(angle-phi(ii))).^2);
    vq=interp1(gamma,Q,gamma2);
    recon(ii)=recon(ii)+dtheta.*(1./(L.^2)).*vq;
    imshow(recon,[]);
end
% recon=recon.*dtheta;
% recon=rot90(recon,2);
imshow(recon,[0 2]);
figure;imshow(recon-P,[]);
figure;
plot(recon(256,:));
hold on;
plot(P(256,:)); 