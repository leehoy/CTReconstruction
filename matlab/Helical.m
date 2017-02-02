path='I:\LowDose\L067\L067\full_DICOM-CT-PD\';
list=dir(strcat(path,'*.dcm'));
%% Read information from dicom file
dicomdict('set','..\..\..\machine learning\LowDose\DICOMCTPDreader\DICOM-CT-PD-dict_v8.txt');
header=dicominfo(strcat(path,list(1).name));
HelicalPitch=header.SpiralPitchFactor; % helical pitch 
Projection2Pi=double(header.NumberofSourceAngularSteps); %Number of projection data in 2 pi
detColNum = double(header.NumberofDetectorColumns); % Ncol, number of detector columns
detRowNum = double(header.NumberofDetectorRows); %Nrow, number of detector rows 
projection=zeros(detColNum,detRowNum,length(list));
detPixSizeCol = double(header.DetectorElementTransverseSpacing); % dcol, detector column width measured at detector surface, mm
detPixSizeRow = double(header.DetectorElementAxialSpacing); % drow, detector row width measured at detector surface, mm

%% Read projection data
for i=1:length(list)
    projection(:,:,i) = double(dicomread(strcat(path,list(i).name))); % uncorrected projection
    projection(:,:,i) = projection(:,:,i)* double(header.RescaleSlope) + double(header.RescaleIntercept);% projection representing line integral of linear attenuation coefficients, double-precision
end
if strcmp(header.DetectorShape=='CYLINDRICAL')
else
end

%% Define reconstruction parameters
SourceToObject=double(header.DetectorFocalCenterRadialDistance);
SourceToDetector=double(header.ConstantRadialDistance);
dalpha=detPixelSizeCol/(SourceToObject*2); % According to Kak, et al.
alpha=0; % Axial angular coordinate of detector
w=0; % vertical coordinate of detector

%% Perform reconstruction
for i=1:NumberOfViews
    SourcePoint=[];
    g1=(g(:,:,i+1)-g(:,:,i))/dtheta;
    g2=(D./(sqrt(D^2+w.^2)))*g1;
end