close all
f=fopen('D:\MachineLearning\IDLReconCompare.dat');
ref=zeros([512 512 200]);
for i=1:200
    img=fread(f,[512 512],'float32');
    ref(:,:,i)=img';
end
fclose(f);
% Recon=zeros([512 512 512]);
% f=fopen('ConeBeamReconstructed.dat');
% for i=1:512
%     img=fread(f,[512 512],'float32');
%     Recon(:,:,i)=img;
% end
% fclose(f);
plot(ref(256,:,100),'r');
hold on
plot(recon(256,:),'k');