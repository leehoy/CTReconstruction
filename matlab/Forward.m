field1='SourceToAxis';
field2='SourceToDetector';
field3='DetectorWidth';
field4='DetectorHeight';
field5='NumberOfRows';
field6='NumberOfChannels';
field7='DetectorPixelSize';
field8='ImagePixelSize';
field9='DetectorCenterOffset';
field10='NumberOfViews';
field11='AngleCoverage';
field12='type';
field13='InitialSource';
field14='DetectorHorizontalVectorOffset';
field15='DetectorVerticalVectorOffset';
field16='ObjectSize';
parameters=struct(field1,[],field2,[],field3,[],field4,[],field5,[],field6,[],...
    field7,[],field8,[],field9,[],field10,[],field11,[],field12,[],field13,[],...
    field14,[],field15,[],field16,[]);
parameters.SourceToAxis=1000;
parameters.SourceToDetector=1500;
parameters.DetectorWidth=400;
parameters.DetectorHeight=300;
parameters.NumberOfRows=1024;
parameters.NumberOfChannels=768;
parameters.DetectorPixelSize=[parameters.NumberOfRows/parameters.DetectorWidth,...
    parameters.NumberOfChannels/parameters.DetectorHeight];
parameters.DetectorCenterOffset=[0,0]; %x, y
parameters.CenterOfSource=[0,0,0];
parameters.ObjectSize=[nx,ny,nz];
parameters.ImagePixelSize=[dx,dy,dz];
% parameters.DetectorNormal=[0,0,1];
parameters.AngleCoverage=2*pi;
parameters.NumberOfViews=100;
parameters.type='distance-driven';
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
