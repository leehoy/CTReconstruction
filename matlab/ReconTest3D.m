% filepath='Proj_distance3D_2.dat';
% filepath='proj_distance_360.dat';
filepath='proj_distance_360_2.dat';
params=struct('SourceToAxis',[],'SourceToDetector',[]','cutoff',[],'FilterType',[],...
    'DetectorPixelSpacing',[],'ReconSpacingX',[],'ReconSpacingY',[],'nx',[],'ny',[],...
    'nu',[],'nv',[],'nw',[]);

%% Geometry parameters
params.SourceToAxis=1000;
params.SourceToDetector=1500;
params.nu=512;
params.nv=384;
params.nview=360;
params.du=0.5;
params.dv=0.5;
% reding projection file need to be implemented

%% Reconstruction parameters
params.nx=256;
params.ny=256;
params.nz=256;
params.ReconSpacingX=0.5;
params.ReconSpacingY=0.5;
params.ReconSpacingZ=0.5;
params.FilterType='ram-lak'; % filter type can be ram-lak, shepp-logan, cosine, hamming, and hann
params.cutoff=1; % cutoff must be posed between 0~1
dtype='float32';

%% Reconstruction method
params.method='distance';
params.direction='ccw'; % this will be included in the future

%% Read projection data
proj=zeros(params.nu,params.nv,params.nview);
f=fopen(filepath);
for i=1:params.nview
    proj(:,:,i)=fread(f,[params.nu,params.nv],dtype);
end
fclose(f);
%% Perform reconstruction
tic;
if(strcmpi(params.method,'distance'))
    recon=DistanceDrivenBackprojection3D(proj,params);
elseif(strcmpi(params.method,'ray'))
    recon=RayDrivenBackprojection3D(proj,params);
elseif(strcmpi(params.method,'pixel'))
    %not implemented yet
else
    fprintf('Error! The selected method is not implemented.\n');
end
toc
