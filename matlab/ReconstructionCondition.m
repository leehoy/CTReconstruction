filepath='';
params=struct('SourceToAxis',[],'SourceToDetector',[]','cutoff',[],'FilterType',[],...
    'DetectorPixelSpacing',[],'ReconSpacingX',[],'ReconSpacingY',[],'nx',[],'ny',[],...
    'nu',[],'nv',[],'nw',[]);

%% Geometry parameters
params.SourceToAxis=1000;
params.SourceToDetector=1500;
params.DetectorPixelSpacing=0.5;
params.nu=512;
params.nv=1;
params.nz=360;
% reding projection file need to be implemented

%% Reconstruction parameters
params.nx=256;
params.ny=256;
params.ReconSpacingX=0.5;
params.ReconSpacingY=-0.5;
params.FilterType='ram-lak'; % filter type can be ram-lak, shepp-logan, cosine, hammin, and hann
params.cutoff=1; % cutoff must be posed between 0~0.5

%% Reconstruction method
params.method='distance';
params.direction='ccw'; % this will be included in the future
tic;
if(strcmpi(params.method,'distance'))
    recon=DistanceDrivenBackprojection2D(proj,params);
elseif(strcmpi(params.method,'ray'))
    recon=RayDrivenBackprojection2D(proj,params);
elseif(strcmpi(params.method,'pixel'))
    %not implemented yet
else
    fprintf('Error! The selected method is not implemented.\n');
end
toc