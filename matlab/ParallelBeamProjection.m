P=phantom('Modified Shepp-Logan',512);
theta=linspace(0,360,180);
[R,xp]=radon(P,theta);
imshow(R,[]);
% f=fopen('D:\workspace\Reconstruction\src\parallel.dat','w');
% fwrite(f,R,'float32');
% fclose(f);