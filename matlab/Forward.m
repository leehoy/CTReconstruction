field1='SourceToAxisDistance';
field2='SourceToDetectorDistance';
field3='DetectorRealSize';%[width height]
field5='NumberOfDetectorPixels'; %[width height]
field7='DetectorPixelSpacing';%[width height]
field9='ImagePixelSpacing';%[dx dy dz]
field10='DetectorCenterOffset';
field11='NumberOfViews';
field12='StartAngle';
field13='EndAngle';% in radian
field14='method';
field15='InitialSource'; %[x y]
field16='InitialDetector';%[x y]
field17='ImageNumberOfPixel';%[nx ny nz]
field18='Dir';%direction of projection angle
parameters=struct(field1,[],field2,[],field3,[],field4,[],field5,[],field6,[],...
    field7,[],field8,[],field9,[],field10,[],field11,[],field12,[],field13,[],...
    field14,[],field15,[],field16,[],field17,[] , field18,[]);
parameters.SourceToAxis=1000;
parameters.SourceToDetector=1500;
parameters.DetectorRealSize=[400,300];
parameters.NumberOfDetectorPixels=[1024,768];
parameters.DetectorPixelSpacing=[parameters.NumberOfDetectorPixels(1)/parameters.DetectorRealSize(1),...
    parameters.NumberOfDetectorPixels(2)/parameters.DetectorRealSize(2)];
parameters.DetectorCenterOffset=[0,0]; %x, y
parameters.InitialSource=[0,0,0];
parameters.ImageNumberOfPixel=[nx,ny,nz];
parameters.ImagePixelSpacing=[dx,dy,dz];
% parameters.DetectorNormal=[0,0,1];
parameters.AngleCoverage=2*pi;
parameters.NumberOfViews=100;
parameters.type='siddon';
nx=512;
ny=512;
nz=430;
filename='D:\DeepLearning\WholeBodyProjection\PediInterpolTotal\Recon.dat';
image=zeros(nx,ny,nz);
f=fopen(filename);
for i=1:nz
    image(:,:,i)=fread(f,[nx ny],'float32');
end
fclose(f);
% ForwardProjection(image,parameters);
