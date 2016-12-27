%% Parameters required for reconstruction
SourceToAxis=1000;
SourceToDetector=1500;
DetectorPixelWidth=0.390625;
DetectorPixelHeight=0.390625;
DetectorWidth=1024;
DetectorHeight=768;
NumberOfViews=680;
AngleCoverage=2*pi;
DataPath='D:\Reconstruction\CBCTData\LogData\';
precision='float32';
ReconX=512;
ReconY=512;
ReconZ=512;
FilterType='hann';
cutoff=0.3;
%% Read projection data from file
list=dir(strcat(DataPath,'*.dat'));
Projection=zeros(DetectorWidth,DetectorHeight,NumberOfViews);
for i=1:NumberOfViews
    f=fopen(strcat(DataPath,list(i).name));
    Projection(:,:,i)=fread(f,[DetectorWidth,DetectorHeight],precision);
    fclose(f);
end
ConeBeamReconstruction;